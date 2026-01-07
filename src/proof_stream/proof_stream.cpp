#include "proof_stream/proof_stream.hpp"
#include "bincode_ffi.hpp"
#include <stdexcept>
#include <iostream>
#include <vector>

namespace triton_vm {
namespace {

uint32_t proof_item_discriminant(ProofItemType type) {
    // Discriminants match Rust's BFieldCodec derive order exactly
    return static_cast<uint32_t>(type);
}

void append_bfield_vector(
    std::vector<BFieldElement>& buffer,
    const std::vector<BFieldElement>& data
) {
    buffer.push_back(BFieldElement(static_cast<uint64_t>(data.size())));
    buffer.insert(buffer.end(), data.begin(), data.end());
}

// For variable-length Vec<XFieldElement> - adds length prefix
void append_xfield_vector(
    std::vector<BFieldElement>& buffer,
    const std::vector<XFieldElement>& data
) {
    buffer.push_back(BFieldElement(static_cast<uint64_t>(data.size())));
    for (const auto& x : data) {
        buffer.push_back(x.coeff(0));
        buffer.push_back(x.coeff(1));
        buffer.push_back(x.coeff(2));
    }
}

// For fixed-length arrays [XFieldElement; N] - NO length prefix
void append_xfield_array(
    std::vector<BFieldElement>& buffer,
    const std::vector<XFieldElement>& data
) {
    for (const auto& x : data) {
        buffer.push_back(x.coeff(0));
        buffer.push_back(x.coeff(1));
        buffer.push_back(x.coeff(2));
    }
}

// For Vec<[BFieldElement; N]> - length prefix for Vec, no prefix for each array
void append_bfield_row_vector(
    std::vector<BFieldElement>& buffer,
    const std::vector<std::vector<BFieldElement>>& rows
) {
    buffer.push_back(BFieldElement(static_cast<uint64_t>(rows.size())));
    for (const auto& row : rows) {
        // No length prefix for the fixed-size array
        buffer.insert(buffer.end(), row.begin(), row.end());
    }
}

// For Vec<[XFieldElement; N]> - length prefix for Vec, no prefix for each array
void append_xfield_row_vector(
    std::vector<BFieldElement>& buffer,
    const std::vector<std::vector<XFieldElement>>& rows
) {
    buffer.push_back(BFieldElement(static_cast<uint64_t>(rows.size())));
    for (const auto& row : rows) {
        // No length prefix for the fixed-size array
        append_xfield_array(buffer, row);
    }
}

void append_digest_vector(
    std::vector<BFieldElement>& buffer,
    const std::vector<Digest>& digests
) {
    // Vec<Digest> encodes as: [vec_len, digest_data...]
    // No per-digest length prefix (Digest is a fixed-size array)
    buffer.push_back(BFieldElement(static_cast<uint64_t>(digests.size())));
    for (const auto& digest : digests) {
        for (size_t i = 0; i < Digest::LEN; ++i) {
            buffer.push_back(digest[i]);
        }
    }
}

} // namespace

// ProofItem factory methods
ProofItem ProofItem::merkle_root(const Digest& root) {
    ProofItem item;
    item.type = ProofItemType::MerkleRoot;
    item.digest = root;
    return item;
}

ProofItem ProofItem::make_log2_padded_height(uint32_t height) {
    ProofItem item;
    item.type = ProofItemType::Log2PaddedHeight;
    item.log2_padded_height_value = height;
    return item;
}

ProofItem ProofItem::out_of_domain_main_row(const std::vector<XFieldElement>& row) {
    ProofItem item;
    item.type = ProofItemType::OutOfDomainMainRow;
    item.xfield_vec = row;
    return item;
}

ProofItem ProofItem::out_of_domain_aux_row(const std::vector<XFieldElement>& row) {
    ProofItem item;
    item.type = ProofItemType::OutOfDomainAuxRow;
    item.xfield_vec = row;
    return item;
}

ProofItem ProofItem::out_of_domain_quotient_segments(const std::vector<XFieldElement>& segments) {
    ProofItem item;
    item.type = ProofItemType::OutOfDomainQuotientSegments;
    item.xfield_vec = segments;
    return item;
}

ProofItem ProofItem::fri_codeword(const std::vector<XFieldElement>& codeword) {
    ProofItem item;
    item.type = ProofItemType::FriCodeword;
    item.xfield_vec = codeword;
    return item;
}

ProofItem ProofItem::fri_polynomial(const std::vector<XFieldElement>& polynomial) {
    ProofItem item;
    item.type = ProofItemType::FriPolynomial;
    item.xfield_vec = polynomial;
    return item;
}

ProofItem ProofItem::fri_response(const FriResponse& response) {
    ProofItem item;
    item.type = ProofItemType::FriResponse;
    item.digests = response.auth_structure;
    item.xfield_vec = response.revealed_leaves;
    return item;
}

ProofItem ProofItem::authentication_structure(const std::vector<Digest>& auth_path) {
    ProofItem item;
    item.type = ProofItemType::AuthenticationStructure;
    item.digests = auth_path;
    return item;
}

ProofItem ProofItem::master_main_table_rows(const std::vector<std::vector<BFieldElement>>& rows) {
    ProofItem item;
    item.type = ProofItemType::MasterMainTableRows;
    // Store rows directly as flattened data (no per-row length prefixes for fixed-size arrays)
    // We'll add the row count during encoding
    for (const auto& row : rows) {
        item.bfield_vec.insert(item.bfield_vec.end(), row.begin(), row.end());
    }
    return item;
}

ProofItem ProofItem::master_aux_table_rows(const std::vector<std::vector<XFieldElement>>& rows) {
    ProofItem item;
    item.type = ProofItemType::MasterAuxTableRows;
    // Store rows directly as flattened XFieldElements
    for (const auto& row : rows) {
        item.xfield_vec.insert(item.xfield_vec.end(), row.begin(), row.end());
    }
    return item;
}

ProofItem ProofItem::quotient_segments_elements(const std::vector<std::vector<XFieldElement>>& segments) {
    ProofItem item;
    item.type = ProofItemType::QuotientSegmentsElements;
    // Store segments directly as flattened XFieldElements
    for (const auto& seg : segments) {
        item.xfield_vec.insert(item.xfield_vec.end(), seg.begin(), seg.end());
    }
    return item;
}

ProofItem ProofItem::decode(const std::vector<BFieldElement>& data) {
    if (data.empty()) {
        throw std::runtime_error("ProofItem::decode: empty data");
    }
    size_t idx = 0;
    auto read_length = [&](const char* ctx) -> size_t {
        if (idx >= data.size()) {
            throw std::runtime_error(std::string("ProofItem::decode: missing length for ") + ctx);
        }
        return static_cast<size_t>(data[idx++].value());
    };

    auto read_bfield_vector = [&](std::vector<BFieldElement>& target, const char* ctx) {
        size_t len = read_length(ctx);
        if (idx + len > data.size()) {
            throw std::runtime_error(std::string("ProofItem::decode: overflow reading ") + ctx);
        }
        target.assign(data.begin() + idx, data.begin() + idx + len);
        idx += len;
    };

    auto read_xfield_vector = [&](std::vector<XFieldElement>& target, const char* ctx) {
        size_t len = read_length(ctx);
        target.clear();
        target.reserve(len);
        for (size_t i = 0; i < len; ++i) {
            if (idx + 3 > data.size()) {
                throw std::runtime_error(std::string("ProofItem::decode: overflow reading ") + ctx);
            }
            target.emplace_back(data[idx], data[idx + 1], data[idx + 2]);
            idx += 3;
        }
    };

    auto read_digest_vector = [&](std::vector<Digest>& target, const char* ctx) {
        size_t len = read_length(ctx);
        target.clear();
        target.reserve(len);
        for (size_t i = 0; i < len; ++i) {
            if (idx + Digest::LEN > data.size()) {
                throw std::runtime_error(std::string("ProofItem::decode: overflow reading ") + ctx);
            }
            std::array<BFieldElement, Digest::LEN> elems{};
            for (size_t j = 0; j < Digest::LEN; ++j) {
                elems[j] = data[idx++];
            }
            target.emplace_back(elems);
        }
    };

    uint64_t discriminant = data[idx++].value();
    ProofItem item;
    item.type = static_cast<ProofItemType>(discriminant);

    switch (item.type) {
        case ProofItemType::MerkleRoot: {
            if (idx + Digest::LEN > data.size()) {
                throw std::runtime_error("ProofItem::decode: truncated Merkle root");
            }
            std::array<BFieldElement, Digest::LEN> elems{};
            for (size_t i = 0; i < Digest::LEN; ++i) {
                elems[i] = data[idx++];
            }
            item.digest = Digest(elems);
            break;
        }
        case ProofItemType::Log2PaddedHeight: {
            if (idx >= data.size()) {
                throw std::runtime_error("ProofItem::decode: missing log2 padded height");
            }
            item.log2_padded_height_value = static_cast<uint32_t>(data[idx++].value());
            break;
        }
        case ProofItemType::OutOfDomainMainRow:
        case ProofItemType::OutOfDomainAuxRow:
        case ProofItemType::OutOfDomainQuotientSegments:
        case ProofItemType::FriCodeword:
        case ProofItemType::FriPolynomial: {
            read_xfield_vector(item.xfield_vec, "x-field vector");
            break;
        }
        case ProofItemType::AuthenticationStructure: {
            read_digest_vector(item.digests, "authentication path");
            break;
        }
        case ProofItemType::MasterMainTableRows: {
            read_bfield_vector(item.bfield_vec, "main rows");
            break;
        }
        case ProofItemType::MasterAuxTableRows:
        case ProofItemType::QuotientSegmentsElements: {
            read_bfield_vector(item.bfield_vec, "metadata");
            read_xfield_vector(item.xfield_vec, "x-field data");
            break;
        }
        case ProofItemType::FriResponse: {
            read_digest_vector(item.digests, "fri auth structure");
            read_xfield_vector(item.xfield_vec, "fri revealed leaves");
            break;
        }
    }

    if (idx != data.size()) {
        throw std::runtime_error("ProofItem::decode: trailing data");
    }
    return item;
}

bool ProofItem::include_in_fiat_shamir_heuristic() const {
    // Matches Rust's proof_items! macro exactly
    switch (type) {
        case ProofItemType::MerkleRoot:
        case ProofItemType::OutOfDomainMainRow:
        case ProofItemType::OutOfDomainAuxRow:
        case ProofItemType::OutOfDomainQuotientSegments:
            return true;
        
        // Items implied by Merkle roots don't need to be hashed
        case ProofItemType::AuthenticationStructure:
        case ProofItemType::MasterMainTableRows:
        case ProofItemType::MasterAuxTableRows:
        case ProofItemType::Log2PaddedHeight:
        case ProofItemType::QuotientSegmentsElements:
        case ProofItemType::FriCodeword:
        case ProofItemType::FriPolynomial:
        case ProofItemType::FriResponse:
            return false;
    }
    return false;
}

std::vector<BFieldElement> ProofItem::encode() const {
    // Use Rust FFI for ALL proof items to ensure 100% binary compatibility
    std::vector<BFieldElement> result;
    
    // Prepare data arrays
    std::vector<uint64_t> bfield_data;
    std::vector<uint64_t> xfield_data;
    std::vector<uint64_t> digest_data;
    uint32_t u32_value = 0;
    
    // Convert data based on type
    switch (type) {
        case ProofItemType::MerkleRoot:
            // MerkleRoot(Digest)
            for (size_t i = 0; i < Digest::LEN; ++i) {
                digest_data.push_back(digest[i].value());
            }
            break;
            
        case ProofItemType::OutOfDomainMainRow:
            // OutOfDomainMainRow(Box<MainRow<XFieldElement>>)
            for (const auto& xfe : xfield_vec) {
                for (size_t i = 0; i < XFieldElement::EXTENSION_DEGREE; ++i) {
                    xfield_data.push_back(xfe.coeff(i).value());
                }
            }
            break;
            
        case ProofItemType::OutOfDomainAuxRow:
            // OutOfDomainAuxRow(Box<AuxiliaryRow>)
            for (const auto& xfe : xfield_vec) {
                for (size_t i = 0; i < XFieldElement::EXTENSION_DEGREE; ++i) {
                    xfield_data.push_back(xfe.coeff(i).value());
                }
            }
            break;
            
        case ProofItemType::OutOfDomainQuotientSegments:
            // OutOfDomainQuotientSegments(QuotientSegments)
            for (const auto& xfe : xfield_vec) {
                for (size_t i = 0; i < XFieldElement::EXTENSION_DEGREE; ++i) {
                    xfield_data.push_back(xfe.coeff(i).value());
                }
            }
            break;
            
        case ProofItemType::AuthenticationStructure:
            // AuthenticationStructure(Vec<Digest>)
            for (const auto& d : digests) {
                for (size_t i = 0; i < Digest::LEN; ++i) {
                    digest_data.push_back(d[i].value());
                }
            }
            break;
            
        case ProofItemType::MasterMainTableRows:
            // MasterMainTableRows(Vec<MainRow<BFieldElement>>)
            for (const auto& bfe : bfield_vec) {
                bfield_data.push_back(bfe.value());
            }
            break;
            
        case ProofItemType::MasterAuxTableRows:
            // MasterAuxTableRows(Vec<AuxiliaryRow>)
            for (const auto& xfe : xfield_vec) {
                for (size_t i = 0; i < XFieldElement::EXTENSION_DEGREE; ++i) {
                    xfield_data.push_back(xfe.coeff(i).value());
                }
            }
            break;
            
        case ProofItemType::Log2PaddedHeight:
            // Log2PaddedHeight(u32)
            u32_value = log2_padded_height_value;
            break;
            
        case ProofItemType::QuotientSegmentsElements:
            // QuotientSegmentsElements(Vec<QuotientSegments>)
            for (const auto& xfe : xfield_vec) {
                for (size_t i = 0; i < XFieldElement::EXTENSION_DEGREE; ++i) {
                    xfield_data.push_back(xfe.coeff(i).value());
                }
            }
            break;
            
        case ProofItemType::FriCodeword:
            // FriCodeword(Vec<XFieldElement>)
            for (const auto& xfe : xfield_vec) {
                for (size_t i = 0; i < XFieldElement::EXTENSION_DEGREE; ++i) {
                    xfield_data.push_back(xfe.coeff(i).value());
                }
            }
            break;
            
        case ProofItemType::FriPolynomial:
            // FriPolynomial(Polynomial<'static, XFieldElement>)
            // Note: Rust's Polynomial::coefficients() trims trailing zeros
            // We need to trim trailing zeros before encoding
            {
                size_t last_non_zero = xfield_vec.size();
                while (last_non_zero > 0 && xfield_vec[last_non_zero - 1].is_zero()) {
                    last_non_zero--;
                }
                for (size_t i = 0; i < last_non_zero; ++i) {
                    for (size_t j = 0; j < XFieldElement::EXTENSION_DEGREE; ++j) {
                        xfield_data.push_back(xfield_vec[i].coeff(j).value());
                    }
                }
            }
            break;
            
        case ProofItemType::FriResponse:
            // FriResponse(FriResponse)
            for (const auto& d : digests) {
                for (size_t i = 0; i < Digest::LEN; ++i) {
                    digest_data.push_back(d[i].value());
                }
            }
            for (const auto& xfe : xfield_vec) {
                for (size_t i = 0; i < XFieldElement::EXTENSION_DEGREE; ++i) {
                    xfield_data.push_back(xfe.coeff(i).value());
                }
            }
            break;
    }
    
    // Call Rust FFI to encode (includes discriminant)
    uint64_t* encoded_data = nullptr;
    size_t encoded_len = 0;
    
    int ffi_result = proof_item_encode_general(
        static_cast<uint32_t>(type),  // discriminant
        bfield_data.empty() ? nullptr : bfield_data.data(),
        bfield_data.size(),
        xfield_data.empty() ? nullptr : xfield_data.data(),
        xfield_data.size() / XFieldElement::EXTENSION_DEGREE,  // number of XFieldElements
        digest_data.empty() ? nullptr : digest_data.data(),
        digest_data.size() / Digest::LEN,  // number of Digests
        u32_value,
        &encoded_data,
        &encoded_len
    );
    
    if (ffi_result != 0 || encoded_data == nullptr) {
        throw std::runtime_error("Failed to encode ProofItem using Rust FFI");
    }
    
    // Copy encoded data to result (FFI result includes discriminant)
    for (size_t i = 0; i < encoded_len; ++i) {
        result.push_back(BFieldElement(encoded_data[i]));
    }
    
    // Free memory allocated by FFI
    proof_item_free_encoding(encoded_data, encoded_len);
    
    return result;
}

// Old encode function removed - now using Rust FFI for all proof items

// try_into methods
bool ProofItem::try_into_merkle_root(Digest& out) const {
    if (type != ProofItemType::MerkleRoot) return false;
    out = digest;
    return true;
}

bool ProofItem::try_into_out_of_domain_main_row(std::vector<XFieldElement>& out) const {
    if (type != ProofItemType::OutOfDomainMainRow) return false;
    out = xfield_vec;
    return true;
}

bool ProofItem::try_into_out_of_domain_aux_row(std::vector<XFieldElement>& out) const {
    if (type != ProofItemType::OutOfDomainAuxRow) return false;
    out = xfield_vec;
    return true;
}

bool ProofItem::try_into_out_of_domain_quotient_segments(std::vector<XFieldElement>& out) const {
    if (type != ProofItemType::OutOfDomainQuotientSegments) return false;
    out = xfield_vec;
    return true;
}

bool ProofItem::try_into_fri_response(FriResponse& out) const {
    if (type != ProofItemType::FriResponse) return false;
    out.auth_structure = digests;
    out.revealed_leaves = xfield_vec;
    return true;
}

bool ProofItem::try_into_master_main_table_rows(std::vector<std::vector<BFieldElement>>& out) const {
    if (type != ProofItemType::MasterMainTableRows) return false;
    // Decode from bfield_vec
    out.clear();
    if (bfield_vec.empty()) return true;
    size_t idx = 0;
    size_t num_rows = static_cast<size_t>(bfield_vec[idx++].value());
    out.reserve(num_rows);
    for (size_t r = 0; r < num_rows; ++r) {
        if (idx >= bfield_vec.size()) return false;
        size_t row_len = static_cast<size_t>(bfield_vec[idx++].value());
        if (idx + row_len > bfield_vec.size()) return false;
        out.emplace_back(bfield_vec.begin() + idx, bfield_vec.begin() + idx + row_len);
        idx += row_len;
    }
    return true;
}

bool ProofItem::try_into_master_aux_table_rows(std::vector<std::vector<XFieldElement>>& out) const {
    if (type != ProofItemType::MasterAuxTableRows) return false;
    out.clear();
    if (bfield_vec.empty()) return true;
    size_t meta_idx = 0;
    size_t x_idx = 0;
    size_t num_rows = static_cast<size_t>(bfield_vec[meta_idx++].value());
    out.reserve(num_rows);
    for (size_t r = 0; r < num_rows; ++r) {
        if (meta_idx >= bfield_vec.size()) return false;
        size_t row_len = static_cast<size_t>(bfield_vec[meta_idx++].value());
        if (x_idx + row_len > xfield_vec.size()) return false;
        out.emplace_back(xfield_vec.begin() + x_idx, xfield_vec.begin() + x_idx + row_len);
        x_idx += row_len;
    }
    return true;
}

bool ProofItem::try_into_quotient_segments_elements(std::vector<std::vector<XFieldElement>>& out) const {
    if (type != ProofItemType::QuotientSegmentsElements) return false;
    out.clear();
    if (bfield_vec.empty()) return true;
    size_t meta_idx = 0;
    size_t x_idx = 0;
    size_t num_segs = static_cast<size_t>(bfield_vec[meta_idx++].value());
    out.reserve(num_segs);
    for (size_t s = 0; s < num_segs; ++s) {
        if (meta_idx >= bfield_vec.size()) return false;
        size_t seg_len = static_cast<size_t>(bfield_vec[meta_idx++].value());
        if (x_idx + seg_len > xfield_vec.size()) return false;
        out.emplace_back(xfield_vec.begin() + x_idx, xfield_vec.begin() + x_idx + seg_len);
        x_idx += seg_len;
    }
    return true;
}

bool ProofItem::try_into_fri_codeword(std::vector<XFieldElement>& out) const {
    if (type != ProofItemType::FriCodeword) return false;
    out = xfield_vec;
    return true;
}

bool ProofItem::try_into_fri_polynomial(std::vector<XFieldElement>& out) const {
    if (type != ProofItemType::FriPolynomial) return false;
    out = xfield_vec;
    return true;
}

bool ProofItem::try_into_log2_padded_height(uint32_t& out) const {
    if (type != ProofItemType::Log2PaddedHeight) return false;
    out = log2_padded_height_value;
    return true;
}

bool ProofItem::try_into_authentication_structure(std::vector<Digest>& out) const {
    if (type != ProofItemType::AuthenticationStructure) return false;
    out = digests;
    return true;
}

// ProofStream implementation
ProofStream::ProofStream()
    : items_index_(0)
    , sponge_(Tip5::init())  // Initialize with fixed-length domain
    , first_varlen_absorption_(true)  // First absorption will switch to variable-length
{
}

ProofStream ProofStream::decode(const std::vector<BFieldElement>& encoding) {
    if (encoding.empty()) {
        throw std::runtime_error("ProofStream::decode: empty encoding");
    }
    size_t idx = 0;
    size_t num_items = static_cast<size_t>(encoding[idx++].value());
    std::vector<ProofItem> items;
    items.reserve(num_items);
    for (size_t i = 0; i < num_items; ++i) {
        if (idx >= encoding.size()) {
            throw std::runtime_error("ProofStream::decode: truncated item length");
        }
        size_t item_len = static_cast<size_t>(encoding[idx++].value());
        if (idx + item_len > encoding.size()) {
            throw std::runtime_error("ProofStream::decode: truncated item payload");
        }
        std::vector<BFieldElement> item_data(
            encoding.begin() + idx,
            encoding.begin() + idx + item_len);
        idx += item_len;
        items.push_back(ProofItem::decode(item_data));
    }
    if (idx != encoding.size()) {
        throw std::runtime_error("ProofStream::decode: trailing data");
    }
    ProofStream stream;
    stream.items_ = std::move(items);
    stream.items_index_ = 0;
    stream.sponge_ = Tip5::init();
    stream.first_varlen_absorption_ = true;
    return stream;
}

void ProofStream::enqueue(const ProofItem& item) {
    static bool debug_enabled = std::getenv("TVM_DEBUG_PROOF_STREAM") != nullptr;
    
    if (debug_enabled) {
        auto encoding = item.encode();
        Tip5 sponge_before = sponge_;
        std::cout << "DEBUG: ProofStream::enqueue - Item #" << items_.size() 
                  << ", type=" << static_cast<int>(item.type)
                  << ", encoding_size=" << encoding.size() << std::endl;
        std::cout << "  Sponge before (first 3): " 
                  << sponge_before.state[0].value() << ","
                  << sponge_before.state[1].value() << ","
                  << sponge_before.state[2].value() << std::endl;
    }
    
    if (item.include_in_fiat_shamir_heuristic()) {
        alter_fiat_shamir_state_with(item.encode());
    }
    items_.push_back(item);
    
    if (debug_enabled) {
        std::cout << "  Sponge after (first 3): " 
                  << sponge_.state[0].value() << ","
                  << sponge_.state[1].value() << ","
                  << sponge_.state[2].value() << std::endl;
        std::cout << std::endl;
    }
}

ProofItem ProofStream::dequeue() {
    if (items_index_ >= items_.size()) {
        throw std::runtime_error("ProofStream: empty queue");
    }
    
    const ProofItem& item = items_[items_index_];
    
    if (item.include_in_fiat_shamir_heuristic()) {
        alter_fiat_shamir_state_with(item.encode());
    }
    
    items_index_++;
    return item;
}

void ProofStream::alter_fiat_shamir_state_with(const std::vector<BFieldElement>& data) {
    pad_and_absorb_all(data);
}

void ProofStream::pad_and_absorb_all(const std::vector<BFieldElement>& elements) {
    // For variable-length input, reset capacity to zeros on first absorption
    // (Tip5::init() sets capacity to ones for fixed-length, but pad_and_absorb_all
    //  handles variable-length data, so we need variable-length mode)
    if (first_varlen_absorption_) {
        for (size_t i = Tip5::RATE; i < Tip5::STATE_SIZE; ++i) {
            sponge_.state[i] = BFieldElement::zero();
        }
        first_varlen_absorption_ = false;
    }
    
    // Pad to multiple of RATE
    std::vector<BFieldElement> padded = elements;
    padded.push_back(BFieldElement::one());  // Padding indicator
    while (padded.size() % Tip5::RATE != 0) {
        padded.push_back(BFieldElement::zero());
    }
    
    // Absorb in chunks of RATE
    for (size_t i = 0; i < padded.size(); i += Tip5::RATE) {
        for (size_t j = 0; j < Tip5::RATE; ++j) {
            sponge_.state[j] = padded[i + j];
        }
        sponge_.permutation();
    }
}

std::array<BFieldElement, Tip5::RATE> ProofStream::squeeze() {
    std::array<BFieldElement, Tip5::RATE> result;
    for (size_t i = 0; i < Tip5::RATE; ++i) {
        result[i] = sponge_.state[i];
    }
    sponge_.permutation();
    return result;
}

std::vector<XFieldElement> ProofStream::sample_scalars(size_t count) {
    std::vector<XFieldElement> result;
    result.reserve(count);
    
    // Each XFieldElement needs 3 BFieldElements
    // Squeeze produces RATE (10) elements at a time
    std::vector<BFieldElement> pool;
    
    while (result.size() < count) {
        // Squeeze more elements if needed
        while (pool.size() < 3) {
            auto squeezed = squeeze();
            for (const auto& elem : squeezed) {
                pool.push_back(elem);
            }
        }
        
        // Take 3 elements for one XFieldElement
        XFieldElement x(pool[0], pool[1], pool[2]);
        result.push_back(x);
        pool.erase(pool.begin(), pool.begin() + 3);
    }
    
    return result;
}

std::vector<XFieldElement> ProofStream::sample_scalars_deterministic(const ChaCha12Rng::Seed& seed, size_t count) {
    ChaCha12Rng rng(seed);
    std::vector<XFieldElement> result;
    result.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        // Each XFieldElement needs 3 coefficients, each as a BFieldElement
        BFieldElement coeff0(rng.next_u64());
        BFieldElement coeff1(rng.next_u64());
        BFieldElement coeff2(rng.next_u64());
        result.emplace_back(coeff0, coeff1, coeff2);
    }

    return result;
}

std::vector<size_t> ProofStream::sample_indices(size_t upper_bound, size_t count) {
    if ((upper_bound & (upper_bound - 1)) != 0) {
        throw std::invalid_argument("upper_bound must be power of 2");
    }
    
    std::vector<size_t> indices;
    indices.reserve(count);
    
    std::vector<BFieldElement> pool;
    
    while (indices.size() < count) {
        // Squeeze more elements if needed
        if (pool.empty()) {
            auto squeezed = squeeze();
            for (auto it = squeezed.rbegin(); it != squeezed.rend(); ++it) {
                pool.push_back(*it);
            }
        }
        
        BFieldElement elem = pool.back();
        pool.pop_back();
        
        // Skip the maximum value to ensure uniform distribution
        static constexpr uint64_t MAX_VALUE = 0xFFFFFFFF00000000ULL;
        if (elem.value() != MAX_VALUE) {
            indices.push_back(static_cast<size_t>(elem.value() % upper_bound));
        }
    }
    
    return indices;
}

std::vector<BFieldElement> ProofStream::encode() const {
    // BFieldCodec for Vec<ProofItem>: length of vector, then for each item:
    //   - length of item encoding (since ProofItem has no static_length)
    //   - item encoding (which includes discriminant + fields)
    // This matches Rust's bfield_codec_encode_list for dynamically-sized items
    std::vector<BFieldElement> encoding;
    encoding.push_back(BFieldElement(static_cast<uint64_t>(items_.size())));
    
    // Debug: Print sizes of each item
    bool debug_proof_sizes = std::getenv("TVM_DEBUG_PROOF_SIZES") != nullptr;
    if (debug_proof_sizes) {
        std::cout << "\n=== PROOF STREAM ENCODING SIZES ===" << std::endl;
        std::cout << "Total items: " << items_.size() << std::endl;
    }
    
    size_t total_item_encoding = 0;
    for (size_t i = 0; i < items_.size(); ++i) {
        const auto& item = items_[i];
        auto item_encoding = item.encode();
        total_item_encoding += item_encoding.size();
        
        if (debug_proof_sizes) {
            std::string item_type = "Unknown";
            switch (item.type) {
                case ProofItemType::MerkleRoot: item_type = "MerkleRoot"; break;
                case ProofItemType::OutOfDomainMainRow: item_type = "OutOfDomainMainRow"; break;
                case ProofItemType::OutOfDomainAuxRow: item_type = "OutOfDomainAuxRow"; break;
                case ProofItemType::OutOfDomainQuotientSegments: item_type = "OutOfDomainQuotientSegments"; break;
                case ProofItemType::AuthenticationStructure: item_type = "AuthenticationStructure"; break;
                case ProofItemType::MasterMainTableRows: item_type = "MasterMainTableRows"; break;
                case ProofItemType::MasterAuxTableRows: item_type = "MasterAuxTableRows"; break;
                case ProofItemType::Log2PaddedHeight: item_type = "Log2PaddedHeight"; break;
                case ProofItemType::QuotientSegmentsElements: item_type = "QuotientSegmentsElements"; break;
                case ProofItemType::FriCodeword: item_type = "FriCodeword"; break;
                case ProofItemType::FriPolynomial: item_type = "FriPolynomial"; break;
                case ProofItemType::FriResponse: item_type = "FriResponse"; break;
            }
            std::cout << "  Item " << i << " (" << item_type << "): " << item_encoding.size() << " elements" << std::endl;
        }
        
        // Prepend length of item encoding (required for dynamically-sized items)
        // The item encoding includes the discriminant, so this length is the total
        encoding.push_back(BFieldElement(static_cast<uint64_t>(item_encoding.size())));
        encoding.insert(encoding.end(), item_encoding.begin(), item_encoding.end());
    }
    
    if (debug_proof_sizes) {
        std::cout << "Total item encoding (without length prefixes): " << total_item_encoding << " elements" << std::endl;
        std::cout << "Total encoding (with length prefixes): " << encoding.size() << " elements" << std::endl;
        std::cout << "  = 1 (Vec length) + " << items_.size() << " (item length prefixes) + " << total_item_encoding << " (item encodings)" << std::endl;
        std::cout << "  = " << (1 + items_.size() + total_item_encoding) << " elements" << std::endl;
        std::cout << "=====================================\n" << std::endl;
    }
    
    return encoding;
}

void ProofStream::encode_and_save_to_file(const std::string& file_path) const {
    // Collect all proof item data for Rust FFI
    const size_t num_items = items_.size();
    
    // Allocate arrays for item data
    std::vector<uint32_t> discriminants(num_items);
    std::vector<const uint64_t*> bfield_data_array(num_items, nullptr);
    std::vector<size_t> bfield_count_array(num_items, 0);
    std::vector<const uint64_t*> xfield_data_array(num_items, nullptr);
    std::vector<size_t> xfield_count_array(num_items, 0);
    std::vector<const uint64_t*> digest_data_array(num_items, nullptr);
    std::vector<size_t> digest_count_array(num_items, 0);
    std::vector<uint32_t> u32_value_array(num_items, 0);
    
    // Storage for item data (keep alive during FFI call)
    std::vector<std::vector<uint64_t>> bfield_data_storage;
    std::vector<std::vector<uint64_t>> xfield_data_storage;
    std::vector<std::vector<uint64_t>> digest_data_storage;
    
    // Collect data for each item
    for (size_t i = 0; i < num_items; ++i) {
        const auto& item = items_[i];
        discriminants[i] = static_cast<uint32_t>(item.type);
        
        // Prepare data arrays based on item type
        switch (item.type) {
            case ProofItemType::MerkleRoot:
                // MerkleRoot(Digest)
                {
                    std::vector<uint64_t> digest_data;
                    for (size_t j = 0; j < Digest::LEN; ++j) {
                        digest_data.push_back(item.digest[j].value());
                    }
                    digest_data_storage.push_back(std::move(digest_data));
                    digest_data_array[i] = digest_data_storage.back().data();
                    digest_count_array[i] = 1;
                }
                break;
                
            case ProofItemType::OutOfDomainMainRow:
                // OutOfDomainMainRow(Box<MainRow<XFieldElement>>)
                {
                    std::vector<uint64_t> xfield_data;
                    for (const auto& xfe : item.xfield_vec) {
                        for (size_t j = 0; j < XFieldElement::EXTENSION_DEGREE; ++j) {
                            xfield_data.push_back(xfe.coeff(j).value());
                        }
                    }
                    xfield_data_storage.push_back(std::move(xfield_data));
                    xfield_data_array[i] = xfield_data_storage.back().data();
                    xfield_count_array[i] = item.xfield_vec.size();
                }
                break;
                
            case ProofItemType::OutOfDomainAuxRow:
                // OutOfDomainAuxRow(Box<AuxiliaryRow>)
                {
                    std::vector<uint64_t> xfield_data;
                    for (const auto& xfe : item.xfield_vec) {
                        for (size_t j = 0; j < XFieldElement::EXTENSION_DEGREE; ++j) {
                            xfield_data.push_back(xfe.coeff(j).value());
                        }
                    }
                    xfield_data_storage.push_back(std::move(xfield_data));
                    xfield_data_array[i] = xfield_data_storage.back().data();
                    xfield_count_array[i] = item.xfield_vec.size();
                }
                break;
                
            case ProofItemType::OutOfDomainQuotientSegments:
                // OutOfDomainQuotientSegments(QuotientSegments)
                {
                    std::vector<uint64_t> xfield_data;
                    for (const auto& xfe : item.xfield_vec) {
                        for (size_t j = 0; j < XFieldElement::EXTENSION_DEGREE; ++j) {
                            xfield_data.push_back(xfe.coeff(j).value());
                        }
                    }
                    xfield_data_storage.push_back(std::move(xfield_data));
                    xfield_data_array[i] = xfield_data_storage.back().data();
                    xfield_count_array[i] = item.xfield_vec.size();
                }
                break;
                
            case ProofItemType::AuthenticationStructure:
                // AuthenticationStructure(Vec<Digest>)
                {
                    std::vector<uint64_t> digest_data;
                    for (const auto& d : item.digests) {
                        for (size_t j = 0; j < Digest::LEN; ++j) {
                            digest_data.push_back(d[j].value());
                        }
                    }
                    digest_data_storage.push_back(std::move(digest_data));
                    digest_data_array[i] = digest_data_storage.back().data();
                    digest_count_array[i] = item.digests.size();
                }
                break;
                
            case ProofItemType::MasterMainTableRows:
                // MasterMainTableRows(Vec<MainRow<BFieldElement>>)
                {
                    std::vector<uint64_t> bfield_data;
                    for (const auto& bfe : item.bfield_vec) {
                        bfield_data.push_back(bfe.value());
                    }
                    bfield_data_storage.push_back(std::move(bfield_data));
                    bfield_data_array[i] = bfield_data_storage.back().data();
                    bfield_count_array[i] = item.bfield_vec.size();
                }
                break;
                
            case ProofItemType::MasterAuxTableRows:
                // MasterAuxTableRows(Vec<AuxiliaryRow>)
                {
                    std::vector<uint64_t> xfield_data;
                    for (const auto& xfe : item.xfield_vec) {
                        for (size_t j = 0; j < XFieldElement::EXTENSION_DEGREE; ++j) {
                            xfield_data.push_back(xfe.coeff(j).value());
                        }
                    }
                    xfield_data_storage.push_back(std::move(xfield_data));
                    xfield_data_array[i] = xfield_data_storage.back().data();
                    xfield_count_array[i] = item.xfield_vec.size();
                }
                break;
                
            case ProofItemType::Log2PaddedHeight:
                // Log2PaddedHeight(u32)
                u32_value_array[i] = item.log2_padded_height_value;
                break;
                
            case ProofItemType::QuotientSegmentsElements:
                // QuotientSegmentsElements(Vec<QuotientSegments>)
                {
                    std::vector<uint64_t> xfield_data;
                    for (const auto& xfe : item.xfield_vec) {
                        for (size_t j = 0; j < XFieldElement::EXTENSION_DEGREE; ++j) {
                            xfield_data.push_back(xfe.coeff(j).value());
                        }
                    }
                    xfield_data_storage.push_back(std::move(xfield_data));
                    xfield_data_array[i] = xfield_data_storage.back().data();
                    xfield_count_array[i] = item.xfield_vec.size();
                }
                break;
                
            case ProofItemType::FriCodeword:
                // FriCodeword(Vec<XFieldElement>)
                {
                    std::vector<uint64_t> xfield_data;
                    for (const auto& xfe : item.xfield_vec) {
                        for (size_t j = 0; j < XFieldElement::EXTENSION_DEGREE; ++j) {
                            xfield_data.push_back(xfe.coeff(j).value());
                        }
                    }
                    xfield_data_storage.push_back(std::move(xfield_data));
                    xfield_data_array[i] = xfield_data_storage.back().data();
                    xfield_count_array[i] = item.xfield_vec.size();
                }
                break;
                
            case ProofItemType::FriPolynomial:
                // FriPolynomial(Polynomial<'static, XFieldElement>)
                // Note: Rust's Polynomial::coefficients() trims trailing zeros
                {
                    size_t last_non_zero = item.xfield_vec.size();
                    while (last_non_zero > 0 && item.xfield_vec[last_non_zero - 1].is_zero()) {
                        last_non_zero--;
                    }
                    std::vector<uint64_t> xfield_data;
                    for (size_t j = 0; j < last_non_zero; ++j) {
                        for (size_t k = 0; k < XFieldElement::EXTENSION_DEGREE; ++k) {
                            xfield_data.push_back(item.xfield_vec[j].coeff(k).value());
                        }
                    }
                    xfield_data_storage.push_back(std::move(xfield_data));
                    xfield_data_array[i] = xfield_data_storage.back().data();
                    xfield_count_array[i] = last_non_zero;
                }
                break;
                
            case ProofItemType::FriResponse:
                // FriResponse(FriResponse)
                {
                    std::vector<uint64_t> digest_data;
                    for (const auto& d : item.digests) {
                        for (size_t j = 0; j < Digest::LEN; ++j) {
                            digest_data.push_back(d[j].value());
                        }
                    }
                    digest_data_storage.push_back(std::move(digest_data));
                    digest_data_array[i] = digest_data_storage.back().data();
                    digest_count_array[i] = item.digests.size();
                    
                    std::vector<uint64_t> xfield_data;
                    for (const auto& xfe : item.xfield_vec) {
                        for (size_t j = 0; j < XFieldElement::EXTENSION_DEGREE; ++j) {
                            xfield_data.push_back(xfe.coeff(j).value());
                        }
                    }
                    xfield_data_storage.push_back(std::move(xfield_data));
                    xfield_data_array[i] = xfield_data_storage.back().data();
                    xfield_count_array[i] = item.xfield_vec.size();
                }
                break;
        }
    }
    
    // Call Rust FFI to encode and serialize
    int result = proof_stream_encode_and_serialize(
        discriminants.data(),
        num_items,
        bfield_data_array.data(),
        bfield_count_array.data(),
        xfield_data_array.data(),
        xfield_count_array.data(),
        digest_data_array.data(),
        digest_count_array.data(),
        u32_value_array.data(),
        file_path.c_str()
    );
    
    if (result != 0) {
        throw std::runtime_error("Failed to encode and serialize proof stream using Rust FFI: " + file_path);
    }
}

} // namespace triton_vm

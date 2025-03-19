module TextUtils

export build_vocabulary, text_to_indices, pad_sequence

function build_vocabulary(texts::Vector{String}, vocab_size::Int)
    word_freq = Dict{String, Int}()
    for text in texts
        for word in split(lowercase(text))
            word_freq[word] = get(word_freq, word, 0) + 1
        end
    end
    sorted_words = sort(collect(keys(word_freq)), by = w -> -word_freq[w])
    vocab = sorted_words[1:min(vocab_size, length(sorted_words))]
    word_to_index = Dict{String, Int}()
    for (i, w) in enumerate(vocab)
        word_to_index[w] = i
    end
    return word_to_index
end

function text_to_indices(text::String, word_to_index::Dict{String,Int})
    return [ get(word_to_index, word, 0) for word in split(lowercase(text)) ]
end

function pad_sequence(indices::Vector{Int}, max_len::Int)
    padded = zeros(Int, max_len)
    len = min(length(indices), max_len)
    padded[1:len] = indices[1:len]
    return padded
end

end  # module TextUtils

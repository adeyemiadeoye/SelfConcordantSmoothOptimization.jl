ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using PythonCall
using DataStructures
using Flux: onehotbatch

function fed_data_to_jl(clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances, num_clients, with_class_completion, prefix_cli)
        """
            Convert federated data from Python objects (as returned by fedartml) to Julia-native structures.

            # Arguments
            - `clients_glob_dic`: Python dictionary containing client data (features and targets).
            - `list_ids_sampled_dic`: Python dictionary of sampled indices for each client.
            - `miss_class_per_node`: Python object with missing class information per node.
            - `distances`: Python dictionary of distances between clients.
            - `num_clients`: Number of clients (Int).
            - `with_class_completion`: Boolean, whether to use the 'with_class_completion' split.
            - `prefix_cli`: String prefix for client keys.

            # Returns
            Tuple of:
            - `clients_data`: Dict mapping client index to (features, targets) tuple.
            - `client_idxs`: Dict mapping client index to list of sample indices.
            - `miss_class_per_node`: Vector of missing class counts per node.
            - `distances_jl`: Dict of distances between clients (as OrderedDicts).
    """

    dict_key = with_class_completion ? "with_class_completion" : "without_class_completion"

    clients_data = Dict()
    client_idxs = Dict()
    for k in 0:num_clients-1
        key_k = prefix_cli*"_$(k+1)"
        size_k = length(list_ids_sampled_dic[dict_key][k])
        features_k = Vector{Array}(undef, size_k)
        targets_k = Vector{Int64}(undef, size_k)
        for i in 0:size_k-1
            feat_i = clients_glob_dic[dict_key][key_k][i][0]
            target_i = clients_glob_dic[dict_key][key_k][i][1]
            features_k[i+1] = pyconvert(Array, feat_i)
            targets_k[i+1] = pyconvert(Int64, target_i)
        end
        n = size(features_k[1], 1)
        d = size(features_k[1], 2)
        features_k = vcat(features_k...)
        features_k = reshape(features_k, (n, d, size_k))
        features_k = permutedims(features_k, (3, 1, 2))
        if size(features_k)[end] == 1
            features_k = reshape(features_k, size(features_k)[1:end-1])
        end

        targets_k = Matrix{Float64}(onehotbatch(targets_k, sort(unique(targets_k)))')
        
        clients_data[k+1] = (features=features_k, targets=targets_k)
        client_idxs[k+1] = pyconvert(Vector, list_ids_sampled_dic[dict_key][k])
    end

    distances_jl = Dict(pyconvert(String, j) => OrderedDict() for j in distances.keys())
    for j in distances.keys()
        for k in distances[j].keys()
            distances_jl[pyconvert(String, j)][pyconvert(String, k)] = pyconvert(Float64, distances[j][k])
        end
    end

    miss_class_per_node = pyconvert(Vector{Int64}, miss_class_per_node)
    

    return clients_data, client_idxs, miss_class_per_node, distances_jl

end

function get_fed_dataset(X::OperatorOrArray2, y::VectorBitVectorOrArray2{<:Real}, num_clients::Int64; method::String="dirichlet", alpha::IntOrFloat=0.5, percent_noniid=0, with_class_completion::Bool=true, prefix_cli::String="Local_node", random_state::Int64=1234)

    splitasfed = pyimport("fedartml").SplitAsFederatedData
    np = pyimport("numpy")

    x_train_glob, y_train_glob = np.array(X), np.array(y)

    # instantiate a SplitAsFederatedData object
    my_federater = splitasfed(random_state=random_state)

    # get federated dataset from centralized dataset
    clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances = my_federater.create_clients(image_list=x_train_glob, label_list=y_train_glob, num_clients=num_clients, prefix_cli=prefix_cli, method=method, alpha=alpha, percent_noniid=percent_noniid)

    # Python to Julia
    clients_data, client_idxs, miss_class_per_node, distances = fed_data_to_jl(clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances, num_clients, with_class_completion, prefix_cli)

    return clients_data, client_idxs, miss_class_per_node, distances
end
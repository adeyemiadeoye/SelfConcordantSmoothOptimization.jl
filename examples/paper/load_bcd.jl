using NPZ

function load_bcd_result(model)
    file_dir = pwd() * "/examples/paper/bcd/"
    m, n = size(model.A)
    sol = vec(npzread(file_dir*"$(m)_$(n)_bcd_sol.npy"))
    obj = model.f(sol) + get_reg(model, sol, "gl")
    mse = npzread(file_dir*"$(m)_$(n)_mses.npy")
    t = npzread(file_dir*"times_sgl_$(m)_$(n).npy")
    num_nz = nnz(sparse(sol))
    return sol, obj, mse, num_nz, t[1]
end
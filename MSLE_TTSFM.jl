using Random, Distributions, Optim, StatsFuns, DataFrames, CSV, LinearAlgebra, HaltonSequences, FLoops, GLM, ForwardDiff, Base.Threads, SpecialFunctions

pdf_N(e, σ) = pdf(Normal(0, σ), e) # define normal density 

function TTNHN_msle(y, x, z, w, α, βx, Cu, βu, Cw, βw, Cv; draws=2^10-1)
    σᵤ = exp.(Cu .+ z * βu)  # parameterize parameter instead of variance
    σₘ = exp.(Cw .+ w * βw)
    σᵥ² = exp(Cv) # scalar
    σᵥ = sqrt(σᵥ²)
    N = length(y)
    ϵ = y .- α .- (x * βx) #(N,)=(N,k)×(k,1)

    # Pre-allocate memory
    llike = Array{Real}(undef, N)
    # Pre-allocate Halton sequences for all observations
    u_halton_seq = collect(Halton(2, length=draws))
    w_halton_seq = collect(Halton(3, length=draws))
    
    @inbounds for i in 1:N
        u_samples = quantile.(truncated(Normal(0, σᵤ[i]), lower=0.0), u_halton_seq)
        w_samples = quantile.(truncated(Normal(0, σₘ[i]), lower=0.0), w_halton_seq)
        diff_samples = ϵ[i] .+ u_samples .- w_samples  
        llike[i] = log(mean(pdf_N.(diff_samples, σᵥ)) + eps())
    end
    
    return -sum(llike) #llike: (N,) --> the sum is a scalar
end

function TTEX_msle(y, x, z, w, α, βx, Cu, βu, Cw, βw, Cv; draws=2^10-1)

    σᵤ = exp.(Cu .+ z * βu)
    σₘ = exp.(Cw .+ w * βw)
    σᵥ² = exp(Cv)
    σᵥ = sqrt(σᵥ²)
    N = length(y)
    ϵ = y .- α .- (x * βx)
    
    # Pre-allocate memory
    llike = Array{Real}(undef, N)
    # Pre-allocate Halton sequences for all observations
    u_halton_seq = collect(Halton(2, length=draws))
    w_halton_seq = collect(Halton(3, length=draws))
    
    @inbounds for i in 1:N
        u_samples = quantile.(Exponential(σᵤ[i]), u_halton_seq)
        w_samples = quantile.(Exponential(σₘ[i]), w_halton_seq)
        diff_samples = ϵ[i] .+ u_samples .- w_samples  
        llike[i] = log(mean(pdf_N.(diff_samples, σᵥ)) + eps()) 
    end
    
    return -sum(llike)
end

function TTWB_msle(y, x, z, w, α, βx, Cu, βu, Cw, βw, shape_u, shape_w, Cv; draws=2^10-1)

    θᵤ = exp.(Cu .+ z * βu)
    θₘ = exp.(Cw .+ w * βw)
    σᵥ² = exp(Cv)
    σᵥ = sqrt(σᵥ²)
    N = length(y)
    ϵ = y .- α .- (x * βx)
    shape_u_pos = exp(shape_u)
    shape_w_pos = exp(shape_w)
    # Pre-allocate memory
    llike = Array{Real}(undef, N)
    # Pre-allocate Halton sequences for all observations
    u_halton_seq = collect(Halton(2, length=draws))
    w_halton_seq = collect(Halton(3, length=draws))

    @inbounds for i in 1:N
        u_samples = quantile.(Weibull(shape_u_pos, θᵤ[i]), u_halton_seq)
        w_samples = quantile.(Weibull(shape_w_pos, θₘ[i]), w_halton_seq)
    
        diff_samples = ϵ[i] .+ u_samples .- w_samples  
        llike[i] = log(mean(pdf_N.(diff_samples, σᵥ)) + eps()) 
    end
    
    return -sum(llike)
end

function TTPT_msle(y, x, z, w, α, βx, Cu, βu, Cw, βw, shape_u, shape_w, Cv; draws=2^10-1)
    θᵤ = exp.(Cu .+ z * βu) 
    θₘ = exp.(Cw .+ w * βw)
    σᵥ² = exp(Cv)
    σᵥ = sqrt(σᵥ²)
    N = length(y)
    ϵ = y .- α .- (x * βx)
    shape_u_pos = exp(shape_u)
    shape_w_pos = exp(shape_w)
    # Pre-allocate memory
    llike = Array{Real}(undef, N)
    # Pre-allocate Halton sequences for all observations
    u_halton_seq = collect(Halton(2, length=draws))
    w_halton_seq = collect(Halton(3, length=draws))
    
    @inbounds for i in 1:N
        # Pareto(shape, scale)
        u_samples = quantile.(Pareto(shape_u_pos, θᵤ[i]), u_halton_seq)
        w_samples = quantile.(Pareto(shape_w_pos, θₘ[i]), w_halton_seq)
    
        diff_samples = ϵ[i] .+ u_samples .- w_samples  
        llike[i] = log(mean(pdf_N.(diff_samples, σᵥ)) + eps()) 
    end
    
    return -sum(llike)
end

# Change to your location

cd("/home5/r12323018/Desktop"); 
df = DataFrame(CSV.File("4Sector_keiretsu_work_data_5.csv")) 
y = df[:, "investment_rate"]       # the dep var
x = Matrix(df[:, ["ln_Tobins_q", "ln_sales_rate", "trend", "D_l", "D_m", "D_s", "D_xs"]]) # the indep vars, not including a constant
z = Matrix(df[:, ["K", "ln_asset"]])
w = Matrix(df[:, ["K", "ln_asset"]])

## setting initial values
# 1. OLS model
ols_model = lm(@formula(investment_rate ~ ln_Tobins_q + ln_sales_rate + trend + D_l + D_m + D_s + D_xs), df)
# 2. SFM model
init = 0.1*ones(15) # 15 coefficients' initial value: α, βx*7, Cu, βu*2, Cw, βw*2, Cv
ols_coeffs = coef(ols_model)[1:8]  # Extract OLS coefficients for α and βx*8
init[1:8] = ols_coeffs            # Use OLS coefficients as initial values of frontier terms

##### Start Estimation #########

func3 = TwiceDifferentiable(vars -> TTEX_msle(y, x, z, w, vars[1], vars[2:8], vars[9], vars[10:11], vars[12], vars[13:14], vars[end], draws=2^10-1),
                            ones(15); autodiff = :forward)

# Step 1: Use Nelder-Mead method for initial optimization
res_nm = optimize(vars -> TTEX_msle(y, x, z, w, vars[1], vars[2:8], vars[9], vars[10:11], vars[12], vars[13:14], vars[end], draws=2^10-1),
                  init, NelderMead(),
                  Optim.Options(g_tol = 1e-3, iterations = 10))

init_newton = Optim.minimizer(res_nm)

# Step 2: Use the result from Nelder-Mead as the initial value for Newton method
res_nt = optimize(func3, init_newton, Newton(), 
                Optim.Options(g_tol = 1e-5, iterations = 50))
                              

# Extract coefficients and handle Hessian matrix
coeff3 = Optim.minimizer(res_nt)

# Use ForwardDiff to calculate Hessian matrix
res_nt_hessian = ForwardDiff.hessian(func3.f, coeff3)

# Symmetrize Hessian if needed
if !issymmetric(res_nt_hessian)
    res_nt_hessian = 0.5 * (res_nt_hessian + res_nt_hessian')
end

# Check positive definiteness
is_pos_def = isposdef(res_nt_hessian) || all(eigvals(res_nt_hessian) .> 0)

#### Hessian Matix Adjusting #####

function make_positive_definite(H; ϵ=1e-6)
    eigenvals, eigenvecs = eigen(Symmetric(H))
    corrected_eigenvals = max.(eigenvals, ϵ)
    return eigenvecs * Diagonal(corrected_eigenvals) * eigenvecs'
end

if !isposdef(res_nt_hessian)
    eigenvals, _ = eigen(Symmetric(res_nt_hessian))
    println("\nThe eigenvalues are ", eigenvals, ". Proceed to the correction but you should take this cautiously.")
    res_nt_hessian = make_positive_definite(res_nt_hessian)
    eigenvals, _ = eigen(Symmetric(res_nt_hessian))
    println("\nThe eigenvalues after the correction are ", eigenvals, ".")
end

is_pos_def = isposdef(res_nt_hessian)

####

# Output results
@show res_nm
@show res_nt
@show is_pos_def
@show res_nt_hessian
TTEX_coeff3 = deepcopy(coeff3)  # Keep coefficients untouched

TTEX_coeff3[end] = exp.(TTEX_coeff3[end])     # convert the last element, log_σᵥ² (Cv) into σᵥ²
_Hessian  = Optim.hessian!(func3, coeff3)  # Hessain evaluated at the coeff vector

var_cov_matrix = inv(_Hessian)

stderror  = sqrt.(diag(var_cov_matrix))

stderror[end] = TTEX_coeff3[end] .*stderror[end]  # convert the unit‘s stderror by using the delta method
t_stats = TTEX_coeff3 ./ stderror

TTEX_table = hcat(TTEX_coeff3, stderror, t_stats)

using PrettyTables
header = ["Coefficients", "Mean", "StdError", "TStatistics"]
row_names = ["α", "log(Tobin's q)", "log(Sales)", "trend", "D_l", "D_m", "D_s", "D_xs",
            "Cu", "Keiretsu", "log(Asset)",
            "Cw", "Keiretsu", "log(Asset)",
            "σᵥ²"]
TTEX_table_with_rows = hcat(row_names, TTEX_table)
pretty_table(TTEX_table_with_rows; header=header, title="Estimation Table")

# DataFrame
df = DataFrame(TTEX_table, header[2:end])
df.Coefficients = row_names
select!(df, "Coefficients", header[2:end])

CSV.write("estimation_TTEX.csv", df)

### 2.  Half-normal

##### Start Estimation #########

func3 = TwiceDifferentiable(vars -> TTNHN_msle(y, x, z, w, vars[1], vars[2:8], vars[9], vars[10:11], vars[12], vars[13:14], vars[end], draws=2^10-1), # α, βx*7, Cu, βu, Cw, βw, Cv
                            ones(15); autodiff = :forward)

# Step 1: Use Nelder-Mead method for initial optimization
res_nm = optimize(vars -> TTNHN_msle(y, x, z, w, vars[1], vars[2:8], vars[9], vars[10:11], vars[12], vars[13:14], vars[end], draws=2^10-1),
                  init, NelderMead(),
                  Optim.Options(g_tol = 1e-3, iterations = 10))

init_newton = Optim.minimizer(res_nm)

# Step 2: Use the result from Nelder-Mead as the initial value for Newton method
res_nt = optimize(func3, init_newton, Newton(), 
                Optim.Options(g_tol = 1e-5, iterations = 50))
                              

# Extract coefficients and handle Hessian matrix
coeff3 = Optim.minimizer(res_nt)

res_nt_hessian = ForwardDiff.hessian(func3.f, coeff3)

# Symmetrize Hessian if needed
if !issymmetric(res_nt_hessian)
    res_nt_hessian = 0.5 * (res_nt_hessian + res_nt_hessian')
end

# Check positive definiteness
is_pos_def = isposdef(res_nt_hessian) || all(eigvals(res_nt_hessian) .> 0)

#### Hessian Matix Adjusting #####

function make_positive_definite(H; ϵ=1e-6)
    eigenvals, eigenvecs = eigen(Symmetric(H))
    corrected_eigenvals = max.(eigenvals, ϵ)
    return eigenvecs * Diagonal(corrected_eigenvals) * eigenvecs'
end

if !isposdef(res_nt_hessian)
    eigenvals, _ = eigen(Symmetric(res_nt_hessian))
    println("\nThe eigenvalues are ", eigenvals, ". Proceed to the correction but you should take this cautiously.")
    res_nt_hessian = make_positive_definite(res_nt_hessian)
    eigenvals, _ = eigen(Symmetric(res_nt_hessian))
    println("\nThe eigenvalues after the correction are ", eigenvals, ".")
end

is_pos_def = isposdef(res_nt_hessian)

####

# Output results
@show res_nm
@show res_nt
@show is_pos_def
@show res_nt_hessian
TTNHN_coeff3 = deepcopy(coeff3)  # Keep coefficients untouched

TTNHN_coeff3[end] = exp.(TTNHN_coeff3[end])     # convert the last element, log_σᵥ² (Cv) into σᵥ²
_Hessian  = Optim.hessian!(func3, coeff3)  # Hessain evaluated at the coeff vector

var_cov_matrix = inv(_Hessian)

stderror  = sqrt.(diag(var_cov_matrix))

stderror[end] = TTNHN_coeff3[end] .*stderror[end]  # convert the unit‘s stderror by using the delta method
t_stats = TTNHN_coeff3 ./ stderror

TTNHN_table = hcat(TTNHN_coeff3, stderror, t_stats)

using PrettyTables
header = ["Variable", "Coefficient", "StdError", "TStatistics"]
row_names = ["α", "log(Tobin's q)", "log(Sales)", "trend", "D_l", "D_m", "D_s", "D_xs",
            "Cu", "Keiretsu", "log(Asset)",
            "Cw", "Keiretsu", "log(Asset)",
            "σᵥ²"]
TTNHN_table_with_rows = hcat(row_names, TTNHN_table)
pretty_table(TTNHN_table_with_rows; header=header, title="Estimation Table")

# DataFrame
df = DataFrame(TTNHN_table, header[2:end])
df.Coefficients = row_names
select!(df, "Coefficients", header[2:end])

CSV.write("estimation_TTNHN.csv", df)

### 3.  Weibull

## setting initial values
ols_model = lm(@formula(investment_rate ~ ln_Tobins_q + ln_sales_rate + trend + D_l + D_m + D_s + D_xs), df)
init = 0.1*ones(17) # now there are 17 variables
ols_coeffs = coef(ols_model)[1:8]
init[1:8] = ols_coeffs

##### Start Estimation #########

func3 = TwiceDifferentiable(vars -> TTWB_msle(y, x, z, w, vars[1], vars[2:8], vars[9], vars[10:11], vars[12], vars[13:14], vars[15], vars[16], vars[end], draws=2^10-1),
                            ones(17); autodiff = :forward)

# Step 1: Use Nelder-Mead method for initial optimization
res_nm = optimize(vars -> TTWB_msle(y, x, z, w, vars[1], vars[2:8], vars[9], vars[10:11], vars[12], vars[13:14], vars[15], vars[16], vars[end], draws=2^10-1),
                  init, NelderMead(),
                  Optim.Options(g_tol = 1e-3, iterations = 10))

init_newton = Optim.minimizer(res_nm)

# Step 2: Use the result from Nelder-Mead as the initial value for Newton method
res_nt = optimize(func3, init_newton, Newton(), 
                Optim.Options(g_tol = 1e-5, iterations = 50))
                              

# Extract coefficients and handle Hessian matrix
coeff3 = Optim.minimizer(res_nt)

# 使用 ForwardDiff 計算 Hessian 矩陣
res_nt_hessian = ForwardDiff.hessian(func3.f, coeff3)

# Symmetrize Hessian if needed
if !issymmetric(res_nt_hessian)
    res_nt_hessian = 0.5 * (res_nt_hessian + res_nt_hessian')
end

# Check positive definiteness
is_pos_def = isposdef(res_nt_hessian) || all(eigvals(res_nt_hessian) .> 0)

#### Hessian Matix Adjusting #####

function make_positive_definite(H; ϵ=1e-6)
    eigenvals, eigenvecs = eigen(Symmetric(H))
    corrected_eigenvals = max.(eigenvals, ϵ)
    return eigenvecs * Diagonal(corrected_eigenvals) * eigenvecs'
end

if !isposdef(res_nt_hessian)
    eigenvals, _ = eigen(Symmetric(res_nt_hessian))
    println("\nThe eigenvalues are ", eigenvals, ". Proceed to the correction but you should take this cautiously.")
    res_nt_hessian = make_positive_definite(res_nt_hessian)
    eigenvals, _ = eigen(Symmetric(res_nt_hessian))
    println("\nThe eigenvalues after the correction are ", eigenvals, ".")
end

is_pos_def = isposdef(res_nt_hessian)

# Output results
@show res_nm
@show res_nt
@show is_pos_def
@show res_nt_hessian
TTWB_coeff3 = deepcopy(coeff3)  # Keep coefficients untouched

TTWB_coeff3[end-2:end] = exp.(TTWB_coeff3[end-2:end])     # convert the last 3 element
_Hessian  = Optim.hessian!(func3, coeff3)  # Hessain evaluated at the coeff vector

function compute_var_cov_matrix(H; λ = 1e-6)
    local var_cov_matrix
    try
        var_cov_matrix = inv(H)
    catch e1
        @warn "inv(Hessian) failed, trying Cholesky decomposition..." exception = e1
        try
            L = cholesky(H)
            var_cov_matrix = L \ I
        catch e2
            @warn "Cholesky failed. Trying regularization H + λI..." exception = e2
            try
                H_reg = H + λ * I
                L = cholesky(H_reg)
                var_cov_matrix = L \ I
            catch e3
                @warn "Regularized Cholesky failed. Trying pseudo-inverse..." exception = e3
                try
                    var_cov_matrix = pinv(H)
                catch e4
                    error("All methods failed. Hessian too ill-conditioned. Model needs to be checked.")
                end
            end
        end
    end
    return var_cov_matrix
end

var_cov_matrix = compute_var_cov_matrix(_Hessian)

stderror  = sqrt.(diag(var_cov_matrix))

stderror[end-2:end] = TTWB_coeff3[end-2:end] .*stderror[end-2:end]  # convert the unit‘s stderror by using the delta method
t_stats = TTWB_coeff3 ./ stderror

TTWB_table = hcat(TTWB_coeff3, stderror, t_stats)

using PrettyTables
header = ["Coefficients", "Mean", "StdError", "TStatistics"]
row_names = ["α", "log(Tobin's q)", "log(Sales)", "trend", "D_l", "D_m", "D_s", "D_xs",
            "Cu", "Keiretsu", "log(Asset)",
            "Cw", "Keiretsu", "log(Asset)",
            "shape_u",
            "shape_w",
            "σᵥ²"]
TTWB_table_with_rows = hcat(row_names, TTWB_table)
pretty_table(TTWB_table_with_rows; header=header, title="Estimation Table")

# DataFrame
df = DataFrame(TTWB_table, header[2:end]) 
df.Coefficients = row_names
select!(df, "Coefficients", header[2:end])

CSV.write("estimation_TTWB.csv", df)

### 4. Pareto

##### Start Estimation #########

func3 = TwiceDifferentiable(vars -> TTPT_msle(y, x, z, w, vars[1], vars[2:8], vars[9], vars[10:11], vars[12], vars[13:14], vars[15], vars[16], vars[end], draws=2^10-1),
                            ones(17); autodiff = :forward)

res_nm = optimize(vars -> TTPT_msle(y, x, z, w, vars[1], vars[2:8], vars[9], vars[10:11], vars[12], vars[13:14], vars[15], vars[16], vars[end], draws=2^10-1),
                  init, NelderMead(),
                  Optim.Options(g_tol = 1e-3, iterations = 10))

init_newton = Optim.minimizer(res_nm)

res_nt = optimize(func3, init_newton, Newton(), 
                Optim.Options(g_tol = 1e-5, iterations = 50))

coeff3 = Optim.minimizer(res_nt)

res_nt_hessian = ForwardDiff.hessian(func3.f, coeff3)

if !issymmetric(res_nt_hessian)
    res_nt_hessian = 0.5 * (res_nt_hessian + res_nt_hessian')
end

is_pos_def = isposdef(res_nt_hessian) || all(eigvals(res_nt_hessian) .> 0)

function make_positive_definite(H; ϵ=1e-6)
    eigenvals, eigenvecs = eigen(Symmetric(H))
    corrected_eigenvals = max.(eigenvals, ϵ)
    return eigenvecs * Diagonal(corrected_eigenvals) * eigenvecs'
end

if !isposdef(res_nt_hessian)
    eigenvals, _ = eigen(Symmetric(res_nt_hessian))
    println("\nThe eigenvalues are ", eigenvals, ". Proceed to the correction but you should take this cautiously.")
    res_nt_hessian = make_positive_definite(res_nt_hessian)
    eigenvals, _ = eigen(Symmetric(res_nt_hessian))
    println("\nThe eigenvalues after the correction are ", eigenvals, ".")
end

is_pos_def = isposdef(res_nt_hessian)

@show res_nm
@show res_nt
@show is_pos_def

TTPT_coeff3 = deepcopy(coeff3)
TTPT_coeff3[end-2:end] = exp.(TTPT_coeff3[end-2:end])
_Hessian = Optim.hessian!(func3, coeff3)

function compute_var_cov_matrix(H; λ = 1e-6)
    local var_cov_matrix
    try
        var_cov_matrix = inv(H)
    catch e1
        @warn "inv(Hessian) failed, trying Cholesky decomposition..." exception = e1
        try
            L = cholesky(H)
            var_cov_matrix = L \ I
        catch e2
            @warn "Cholesky failed. Trying regularization H + λI..." exception = e2
            try
                H_reg = H + λ * I
                L = cholesky(H_reg)
                var_cov_matrix = L \ I
            catch e3
                @warn "Regularized Cholesky failed. Trying pseudo-inverse..." exception = e3
                try
                    var_cov_matrix = pinv(H)
                catch e4
                    error("All methods failed. Hessian too ill-conditioned. Model needs to be checked.")
                end
            end
        end
    end
    return var_cov_matrix
end

var_cov_matrix = compute_var_cov_matrix(_Hessian)
stderror = sqrt.(diag(var_cov_matrix))
stderror[end-2:end] = TTPT_coeff3[end-2:end] .* stderror[end-2:end]
t_stats = TTPT_coeff3 ./ stderror

TTPT_table = hcat(TTPT_coeff3, stderror, t_stats)

using PrettyTables
header = ["Variable", "Coefficient", "StdError", "TStatistics"]
row_names = ["α", "log(Tobin's q)", "log(Sales)", "trend", "D_l", "D_m", "D_s", "D_xs",
            "Cu", "Keiretsu", "log(Asset)",
            "Cw", "Keiretsu", "log(Asset)",
            "shape_u", "shape_w", "σᵥ²"]

TTPT_table_with_rows = hcat(row_names, TTPT_table)
pretty_table(TTPT_table_with_rows; header=header, title="Pareto SFA Estimation")

df_output = DataFrame(TTPT_table, header[2:end])
df_output.Variable = row_names
select!(df_output, "Variable", header[2:end])
CSV.write("estimation_TTPT.csv", df_output)
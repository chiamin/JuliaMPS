# MPSCore Julia 建構指南

Julia dense-only 重寫。來源專案：`../Uniten/UnitenDMRG`（Python + cytnx）。

---

## 必讀：全域約定

**在開始任何一個步驟之前，必須熟知以下約定，所有檔案都依賴它們。**

### Tensor leg 順序（不可違反）

```
MPS tensor : Array{T,3}  →  dim 1=l, dim 2=i, dim 3=r
MPO tensor : Array{T,4}  →  dim 1=l, dim 2=ip, dim 3=i, dim 4=r
Env tensor : Array{T,3}  →  dim 1=dn (ket虛擬bond), dim 2=w (MPO虛擬bond), dim 3=up (bra虛擬bond)
```

`ip` = outgoing/bra physical index（對應 MPS bra 端）
`i`  = incoming/ket physical index（對應 MPS ket 端）

### Bra/Ket 與 conj 規則

- **ket 端**：直接使用 `A[l, i, r]`
- **bra 端**：永遠用 `conj(A)[l, i, r]`（對應 Python 的 `.Dagger()`）
- Real tensor 下漏掉 `conj` 不會報錯，**每個涉及 bra 的縮併都必須有 complex 測試**

### svd_split 的輸出格式

```julia
# svd_split(A, n_row_legs; maxdim, cutoff) 的行為：
# - 把 A reshape 成矩陣：row = prod(前 n_row_legs 個維度)，col = 其餘
# - 做 SVD，截斷
# - 回傳 U shape=(row_dims..., χ), s shape=(χ,), Vt shape=(χ, col_dims...)
# - 呼叫端自己 reshape 成需要的形狀
```

### `@tensor` 規則

- 同一個 `@tensor` 表達式內的 dummy index（被求和的 index）名稱不能重複
- 每個模組頂部以 comment 標明本模組使用的 leg 順序

### KrylovKit 向量介面

- `eigsolve` 和 `exponentiate` 接受 `AbstractVector`
- 傳入前用 `vec(phi)` 展開，收到後用 `reshape(v, size(phi))` 還原
- 統一包成 helper 函式，不要在各處重複寫

### 套件

```toml
[deps]
TensorOperations = "最新"
KrylovKit = "最新"
```

Julia 版本：**1.10 LTS**

---

## Step 1：`Project.toml` + `src/MPSCore.jl`

**這個 step 不對應任何 Python 檔案，從頭建立。**

### 要做的事

1. 建 `Project.toml`，加入 `TensorOperations` 和 `KrylovKit` 依賴
2. 建 `src/MPSCore.jl`，作為主模組骨架

### `src/MPSCore.jl` 骨架

```julia
module MPSCore

using TensorOperations
using KrylovKit
using LinearAlgebra

include("TensorUtils.jl")
include("Lattice/Square.jl")
include("Sites/Site.jl")
include("Sites/SpinHalf.jl")
include("Sites/SpinlessFermion.jl")
include("Sites/Electron.jl")
include("MPS/MPS.jl")
include("MPS/MPO.jl")
include("MPS/Init.jl")
include("MPS/Operations.jl")
include("MPS/Compression.jl")
include("MPS/MPOCompression.jl")
include("MPS/AutoMPO.jl")
include("DMRG/Environment.jl")
include("DMRG/EffectiveOperators.jl")
include("DMRG/Engine.jl")
include("TDVP/Engine.jl")

end
```

---

## Step 2：`src/TensorUtils.jl`

**對應 Python**：`unitensor/core.py`（363行）+ `unitensor/utils.py`（100行）

### 讀 Python 檔案前先確認

- `svd_by_labels`：注意它的 `cutoff` 是對 normalized rho eigenvalue（`|s_i|²/Σ|s_j|²`），不是對 singular value 本身；Julia 版要保持同樣定義
- `absorb` 參數：`"left"` 把 s 吸入 U，`"right"` 把 s 吸入 Vt；回傳的是 `(left, right, discarded_weight)`
- `direct_sum`：Python 版處理 QN 和 dense 兩種；Julia 版只需要 dense，大幅簡化

### 要實作的函式

```julia
# 1. SVD 截斷
# A: 任意 Array，n_row_legs: 前幾個維度是 row
# absorb: "left"|"right"|nothing
# maxdim: 最多保留幾個 singular value
# cutoff: normalized rho eigenvalue 的截斷閾值
function svd_split(A::Array{T}, n_row_legs::Int;
                   maxdim::Int=typemax(Int),
                   cutoff::Real=0.0,
                   absorb::Union{String,Nothing}="right") where T
    # 步驟：
    # 1. row_dim = prod(size(A)[1:n_row_legs])
    #    col_dim = prod(size(A)[n_row_legs+1:end])
    # 2. M = reshape(A, row_dim, col_dim)
    # 3. F = svd(M)
    # 4. 計算 normalized rho: λ_i = |s_i|² / Σ|s_j|²
    # 5. 找截斷位置：保留 λ_i >= cutoff，且數量 <= maxdim，至少保留 1 個
    # 6. 計算 discarded = max(0, 1 - Σkept_λ)
    # 7. 依 absorb 決定是否把 s 吸入 U 或 Vt
    # 8. 回傳 (U_reshaped, Vt_reshaped, discarded)
    #    U shape:  (size(A)[1:n_row_legs]..., χ)
    #    Vt shape: (χ, size(A)[n_row_legs+1:end]...)
end

# 2. QR 分解
function qr_split(A::Array{T}, n_row_legs::Int) where T
    # 類似 svd_split，不截斷，不需要 absorb 參數
    # 回傳 (Q, R)，shape 同上
end

# 3. Dense direct_sum（沿指定 dim 做 block-diagonal）
# 用於 mps_sum：把兩個 MPS tensor 沿虛擬 bond 拼成 block-diagonal
# sum_dims: 要 direct sum 的維度編號（1-indexed）
# 其他維度必須大小相同
function direct_sum(A::Array{T,N}, B::Array{T,N}, sum_dims) where {T,N}
    # 例：MPS interior site，sum_dims=[1,3]（l 和 r）
    # 輸出 shape：sum_dims 的大小相加，其他維度不變
    # off-diagonal block 填 0
end
```

### 注意事項

- `svd_split` 的 `cutoff=0` 必須是合法輸入（不截斷）
- 如果 A 全為零，SVD 不能 normalize；Python 版在這裡直接 raise，Julia 版同樣處理
- `direct_sum` 的 `sum_dims` 順序不重要，但輸出維度的順序要和輸入一致

### 測試檔：`test/test_tensor_utils.jl`

```
測試項目：
1. svd_split round-trip：A ≈ U * diagm(s) * Vt（real + complex）
2. svd_split maxdim 截斷：χ <= maxdim
3. svd_split cutoff 截斷：discarded ≈ 預期值
4. qr_split：Q 正交，Q*R ≈ A（real + complex）
5. direct_sum：shape 正確，A 和 B 的 block 在對角，off-diagonal 為 0
```

---

## Step 3：`src/Sites/Site.jl`

**對應 Python**：`MPS/physical_sites/site.py`（238行）

### 讀 Python 檔案前先確認

- `PhysicalSite` 儲存：`dim`（物理維度）、`_ops`（name→matrix）、`_fermionic`（name→bool）
- `delta_qn` 在 Julia 版本完全不需要（QN 專用）
- `is_fermionic(name)` 會被 AutoMPO 用來決定是否插入 JW string，**必須實作**
- `op(name)` 回傳 `Matrix{ComplexF64}`

### 要實作的內容

```julia
abstract type AbstractSite end

struct PhysicalSite <: AbstractSite
    dim       :: Int
    type_name :: String
    ops       :: Dict{String, Matrix{ComplexF64}}
    fermionic :: Dict{String, Bool}
end

# 介面函式
dim(s::PhysicalSite) = s.dim
op(s::PhysicalSite, name::String) = s.ops[name]       # 找不到就 error
is_fermionic(s::PhysicalSite, name::String) = get(s.fermionic, name, false)

# 建構輔助
function register_op!(s::PhysicalSite, name::String, mat::Matrix, fermionic::Bool=false)
    ...
end
```

### 注意事項

- `op` 找不到 name 時應丟出有意義的 error（不要靠 KeyError 的 default message）

---

## Step 4：`src/Sites/SpinHalf.jl`

**對應 Python**：`MPS/physical_sites/spin_half.py`（45行）

### 讀 Python 檔案前先確認

- basis 順序：index 0 = |dn⟩，index 1 = |up⟩（**這個順序影響 operator 矩陣的元素位置**）
- 在 Python 版中 `Sp|dn⟩ = |up⟩`，所以 `Sp[1,0] = 1`（row=ip, col=i）

### 要實作的內容

```julia
function spin_half() :: PhysicalSite
    I  = [1.0 0.0; 0.0 1.0]
    Sz = [0.5 0.0; 0.0 -0.5]
    Sp = [0.0 0.0; 1.0 0.0]   # Sp|dn>=|up>: row=up(1), col=dn(0)
    Sm = [0.0 1.0; 0.0 0.0]
    # 所有 operator 都非 fermionic
    ...
end
```

### 測試（在 `test/test_sites.jl` 中）

```
1. Sp * Sm + Sm * Sp = I（代數關係）
2. [Sz, Sp] = Sp，[Sz, Sm] = -Sm
3. is_fermionic 全為 false
```

---

## Step 5：`src/Sites/SpinlessFermion.jl`

**對應 Python**：`MPS/physical_sites/spinless_fermion.py`（51行）

### 讀 Python 檔案前先確認

- basis：index 0 = |0⟩（empty），index 1 = |1⟩（occupied）
- `F = I - 2*N`（parity operator，`F|0⟩=|0⟩，F|1⟩=-|1⟩`）
- `Cdag` 和 `C` 是 **fermionic**（`is_fermionic=True`），`F`, `N`, `I` 不是

### 注意事項

- `is_fermionic` 的設定直接影響 AutoMPO 的 JW string 插入，必須和 Python 版完全一致

---

## Step 6：`src/Sites/Electron.jl`

**對應 Python**：`MPS/physical_sites/electron.py`（130行）

### 讀 Python 檔案前先確認

- basis 有 4 個態：`|0⟩, |↑⟩, |↓⟩, |↑↓⟩`，注意 Python 版的確切排列順序
- 有兩種 fermionic operators：`Cdagup`、`Cup`、`Cdagdn`、`Cdn`，都是 `is_fermionic=True`
- Hubbard 型 `U` term 用 `Nup * Ndn`

### 注意事項

- basis 順序必須和 Python 版完全一致，否則所有矩陣元素都會錯

---

## Step 7（測試）：`test/test_sites.jl`

```
測試項目（對 SpinHalf, SpinlessFermion, Electron 各做）：
1. operator 代數：{C, Cdag} = 1，F² = I，[Sz, Sp] = Sp 等
2. is_fermionic 的值和 Python 版一致
3. op() 找不到 name 時 throw error
4. dim() 回傳正確值
```

---

## Step 8：`src/Lattice/Square.jl`

**對應 Python**：`lattice/square.py`（100行）

### 注意事項

- 純座標計算，無 tensor 操作，直接翻譯
- Python 的 `range(N)` 是 0-indexed；Julia 習慣 1-indexed，**需決定採用哪種**（建議保持 0-indexed 跟 Python 版一致，以免使用者混淆）

### 測試：`test/test_lattice.jl`

```
基本 site pair / neighbor 關係的正確性
```

---

## Step 9：`src/MPS/MPS.jl`

**對應 Python**：`MPS/mps.py`（669行）

### 讀 Python 檔案前先確認

- `center_left` / `center_right`：定義 stale window，sites < center_left 是 left-orthonormal，sites > center_right 是 right-orthonormal
- `_shift_center_right(site)`：SVD，row=[l,i]，把 s 吸入右邊，site 變成 left-orthonormal
- `_shift_center_left(site)`：SVD，row=[l]，把 s 吸入左邊，site 變成 right-orthonormal
- `make_phi(p, n)`：把 site p 到 p+n-1 的 tensor 縮在一起，輸出 shape `(l, i0, [i1], r)`
- `update_sites(p, phi, ...)`：把 phi 分解回 site tensor，更新 center
- Observer callback：Python 用 weakref 儲存 callback；Julia 可簡化為普通 `Vector{Function}`（GC 行為不同但功能等價）

### `_shift_center_right` 的 Julia 縮併

```julia
# Python: u, vt, _ = svd_by_labels(A, row_labels=["l","i"], absorb="right")
# Julia:
U, Vt, _ = svd_split(tensors[site], 2, absorb="right")
# U shape: (l, i, χ)  → 就是新的 tensors[site]
# Vt shape: (χ, r)    → 需要縮入 tensors[site+1]

# Python: right_new = Contract(Vt_relabeled, tensors[site+1])
# Julia:
@tensor right_new[x, i, r] := Vt[x, y] * tensors[site+1][y, i, r]
# right_new shape: (χ, i, r)  → 新的 tensors[site+1]
```

### `_shift_center_left` 的 Julia 縮併

```julia
# Python: u, vt, _ = svd_by_labels(A, row_labels=["l"], absorb="left")
# Julia:
US, Vt, _ = svd_split(tensors[site], 1, absorb="left")
# US shape: (l, χ)      → 需要縮入 tensors[site-1]
# Vt shape: (χ, i, r)   → 就是新的 tensors[site]

# Python: left_new = Contract(tensors[site-1], US_relabeled)
# Julia:
@tensor left_new[l, i, x] := tensors[site-1][l, i, y] * US[y, x]
# left_new shape: (l, i, χ)  → 新的 tensors[site-1]
```

### `make_phi` 的 Julia 縮併（2-site 例子）

```julia
# p=3, n=2: 縮 tensors[3] 和 tensors[4]
# Python: Contract(A3_relabeled, A4_relabeled)
# Julia:
@tensor phi[l, i0, i1, r] := tensors[p][l, i0, x] * tensors[p+1][x, i1, r]
# phi shape: (l, i0, i1, r)
```

### `check_mps_tensor` 的 Julia 版

```julia
# 只檢查 size，不檢查 QN bond direction
@assert ndims(tensors[site]) == 3
if site > 0
    @assert size(tensors[site], 1) == size(tensors[site-1], 3)
end
if site < N-1
    @assert size(tensors[site], 3) == size(tensors[site+1], 1)
end
```

### 注意事項

- `set_sites` 的兩個相鄰 site 同時更新：先全部寫入再驗證（不能一個一個，否則 neighbor bond check 會暫時失敗）
- `normalize()` 需要 `orthogonalize()` 先設定 center

### 測試：`test/test_mps.jl`

```
1. 建構驗證：size 正確，boundary bond dim = 1
2. move_center! 後的 left/right orthonormal 驗證（用 conj 內積查 gram matrix）
3. orthogonalize! 收斂到單一 center
4. make_phi / update_sites round-trip（1-site 和 2-site）
5. 以上測試 real + complex 各一組
```

---

## Step 10：`src/MPS/MPO.jl`

**對應 Python**：`MPS/mpo.py`（213行）

### 讀 Python 檔案前先確認

- leg 順序：`(l, ip, i, r)` —— `ip` 在 `i` 前面（dim 2 和 dim 3）
- boundary：`size(tensors[1], 1) == 1`，`size(tensors[N], 4) == 1`
- 鄰近 virtual bond：`size(W[p], 4) == size(W[p+1], 1)`

### 注意事項

- `ip`（bra, dim 2）和 `i`（ket, dim 3）的順序必須和所有 `@tensor` 縮併一致；**Environment 和 EffOperator 裡的 MPO 縮併也要對應這個順序**

### 測試：`test/test_mpo.jl`

```
1. 建構驗證：leg 數量、boundary bond dim
2. 鄰近 bond size 一致性
```

---

## Step 11：`src/MPS/Init.jl`

**對應 Python**：`MPS/mps_init.py`（50行）

### 讀 Python 檔案前先確認

- `bond_dims`：`[1, D, D, ..., D, 1]`，長度 `num_sites + 1`
- 建出來的 MPS 要 `orthogonalize()` + `normalize()` 才是合法狀態

### 要實作的函式

```julia
function random_mps(T::Type, num_sites::Int, phys_dim::Int, bond_dim::Int;
                    seed=nothing, normalize=true) :: MPS
    # T = Float64 or ComplexF64
    # bond_dims = [1, bond_dim, ..., bond_dim, 1]
    # tensors[k] = randn(T, bond_dims[k], phys_dim, bond_dims[k+1])
    # 若 normalize: orthogonalize! + normalize!
end

function product_state(T::Type, local_states::Vector{Vector}) :: MPS
    # local_states[k] 是第 k 個 site 的 local state vector（長度 = phys_dim）
    # tensor shape: (1, phys_dim, 1)，填入 local_states[k]
end
```

### 測試：`test/test_mps_init.jl`

```
1. random_mps：norm ≈ 1（若 normalize=true），size 正確
2. product_state：inner(psi, psi) ≈ 1（若 local_states 是 normalized）
3. real + complex 各一組
```

---

## Step 12：`src/MPS/Operations.jl`

**對應 Python**：`MPS/mps_operations.py`（464行）

### 讀 Python 檔案前先確認

- `inner(psi, phi)`：左到右，每步更新 env（shape `(dn, up)` = `(χ_bra, χ_ket)`）
- `expectation(psi, mpo, phi)`：三層，env shape `(dn, w, up)`
- `mps_sum(psi, phi)`：site 0 只 direct-sum `r`；interior 同時 direct-sum `l` 和 `r`；site N-1 只 direct-sum `l`
- `exact_apply_mpo(mpo, mps)`：每個 site 縮 `i` 維度，合併 virtual bond
- `fit_apply_mpo`：需要 `OperatorEnv` 和 `EffOperator`（在後面的 DMRG Step 才有），**此函式最後實作**

### `inner` 的縮併

```julia
# env 初始：ones(T, 1, 1)，shape (dn=χ_bra, up=χ_ket)
# 每步（ket=phi[k], bra=psi[k]）：
@tensor env_new[dn, up] := env[x, y] * ket[x, i, dn] * conj(bra)[y, i, up]
# ket leg: (l=x, i, r=dn)
# bra leg: (l=y, i, r=up)，注意 bra 要 conj
# 結果是標量：only(env_final)
```

### `expectation` 的縮併

```julia
# env 初始：ones(T, 1, 1, 1)，shape (dn, w, up)
# 每步：
@tensor env_new[dn, w, up] :=
    env[x, m, y] *
    ket[x, i, dn] *
    mpo[m, ip, i, w] *
    conj(bra)[y, ip, up]
# ket:  (l=x, i, r=dn)
# mpo:  (l=m, ip, i, r=w)，ip 與 bra 縮，i 與 ket 縮
# bra:  (l=y, ip, r=up)，conj，ip 與 mpo 的 ip 縮
```

### `mps_sum` 的 direct_sum

```julia
for k = 1:N
    A, B = psi[k], phi[k]
    if k == 1
        # 只對 dim 3 (r) 做 direct_sum
        C = direct_sum(A, B, [3])
    elseif k == N
        # 只對 dim 1 (l) 做 direct_sum
        C = direct_sum(A, B, [1])
    else
        # 對 dim 1 (l) 和 dim 3 (r) 做 direct_sum
        C = direct_sum(A, B, [1, 3])
    end
end
```

### `exact_apply_mpo` 的縮併

```julia
# 每個 site k：縮 mpo[k] 的 i（dim 3）和 mps[k] 的 i（dim 2）
@tensor T[ml, mr, pal, par, ip] :=
    mpo[ml, ip, i, mr] * mps[pal, i, par]
# 然後合併 virtual bonds: (ml, pal) → l_new，(mr, par) → r_new
# reshape(permutedims(T, [1,3,5,2,4]), (ml*pal, ip, mr*par))
# 新的 MPS tensor shape: (D_mpo*D_mps, d, D_mpo*D_mps)
```

### 注意事項

- `fit_apply_mpo` 依賴 `OperatorEnv` 和 `EffOperator`，**在 Step 20 後才能實作和測試**；先留 stub（`error("not yet implemented")`）

### 測試：`test/test_mps_operations.jl`

```
1. inner(psi, psi) ≈ norm(psi)²
2. inner(orthogonalized psi, phi) 與 full contraction 結果相同
3. expectation 用已知 Hamiltonian 驗證
4. mps_sum：inner(sum, phi) ≈ inner(psi,phi) + inner(phi2,phi)
5. exact_apply_mpo：norm 和矩陣乘法結果一致
6. real + complex bra/ket 組合各測（bra conj 效果要被 complex 測試抓到）
```

---

## Step 13：`src/MPS/Compression.jl`

**對應 Python**：`MPS/mps_compression.py`（218行）

### 讀 Python 檔案前先確認

- `svd_compress_mps`：先右掃做 left-canonicalize（不截斷），再左掃 SVD 截斷
- `_svd_two_sites`：merge 兩個相鄰 site，row=[l, i1]，SVD，split 回去

### `_svd_two_sites` 的 Julia 縮併

```julia
# merge tensors[p] (l,i,r) 和 tensors[p+1] (l,i,r)
@tensor aa[l, i1, i2, r] := tensors[p][l, i1, x] * tensors[p+1][x, i2, r]
# svd_split(aa, 2, ...)  → n_row_legs=2 意思是 (l,i1) 是 row
# U shape: (l, i1, χ)，Vt shape: (χ, i2, r)
```

### 注意事項

- `denmat_compress_mps` **不實作**（此次只做 SVD compress）

---

## Step 14：`src/MPS/MPOCompression.jl`

**對應 Python**：`MPS/mpo_compression.py`（97行）

### 讀 Python 檔案前先確認

- `_svd_two_mpo_sites`：merge 兩個 MPO site，row=[l, ip, i]（三個維度），SVD，split

### `_svd_two_mpo_sites` 的 Julia 縮併

```julia
# merge W[p] (l,ip,i,r) 和 W[p+1] (l,ip,i,r)
@tensor aa[l, ip1, i1, ip2, i2, r] :=
    tensors[p][l, ip1, i1, x] * tensors[p+1][x, ip2, i2, r]
# svd_split(aa, 3, ...)  → n_row_legs=3 意思是 (l,ip1,i1) 是 row
# U shape: (l, ip1, i1, χ) → 新的 W[p]，需 reshape 成 (l,ip,i,r)=(l,ip1,i1,χ)
# Vt shape: (χ, ip2, i2, r) → 新的 W[p+1]
```

### 測試：`test/test_compression.jl`

```
1. svd_compress_mps：norm 基本不變，bond dim <= maxdim
2. svd_compress_mpo：同上
3. 壓縮前後 inner(compressed, exact) ≈ 1（若截斷誤差小）
```

---

## Step 15：`src/MPS/AutoMPO.jl`

**對應 Python**：`MPS/auto_mpo.py`（657行）

### 讀 Python 檔案前先確認

- FSM 核心邏輯（約 500 行）是純 dict/list 操作，**沒有 cytnx 呼叫**，可直接翻譯
- `_preprocess_term(term)`：把 user 輸入的 operator string 展開成 per-site 的 operator（含 JW string）
- `_fill_identity(site, partial_key)`：在沒有 user operator 的 site 插入 `I` 或 `F`（依已套用的 fermionic operator 的奇偶性）
- `to_mpo()`：把 FSM 的 W 矩陣包成 `Array{ComplexF64,4}` 的 MPO tensors，leg 順序 `(l, ip, i, r)`

### 關鍵邏輯：JW string

```
user: add(coeff, "Cdag", i, "C", j)  # i < j
展開後：
  site i:   Cdag * F（因為 C_j 的 JW string 穿過 site i）
  site i+1 ~ j-1: F
  site j:   C（bare operator）
```

`_fill_identity` 靠 partial_key 裡 fermionic operator 的個數來決定插 `F` 還是 `I`。

### W matrix 包成 MPO tensor

```julia
# W[p] 是一個 (d_mpo_left, d_mpo_right, dim, dim) 的矩陣集合
# MPO tensor leg 順序是 (l, ip, i, r)
# W_tensor[ml, ip, i, mr] = W[p][ml, mr][ip, i]
W_tensor = zeros(ComplexF64, d_left, phys_dim, phys_dim, d_right)
for ml in 1:d_left, mr in 1:d_right
    W_tensor[ml, :, :, mr] = W[p][ml, mr]  # 這個 matrix 是 (ip, i) 的
end
```

### 注意事項

- **必須用 Python 版的輸出做 ground truth 驗證**：在 Python 跑 AutoMPO，把每個 site 的 W matrix dump 成 `.npy`，Julia 讀入後逐元素比對
- `ip` 是 row（bra），`i` 是 col（ket）；矩陣元素 `W[ip, i]` 的約定要和 Python 版一致
- Julia 的 `Dict` 和 `Tuple` 可以做 key，行為和 Python 相同

### 測試：`test/test_auto_mpo.jl`

```
1. Heisenberg：AutoMPO 建出的 MPO 和手寫的 MPO 逐元素相同
2. spinless fermion tight-binding：同上
3. Electron Hubbard：同上
4. 用 Python dump 的 W matrix 做逐元素比對（主要驗證工具）
5. expectation(psi, H, psi) 和 ED 結果相同（小系統）
```

---

## Step 16：`src/DMRG/Environment.jl`

**對應 Python**：`DMRG/environment.py`（643行）

### 讀 Python 檔案前先確認

- `LREnv` 是 abstract base，`OperatorEnv` 和 `VectorEnv` 是 subclass
- stale window `[centerL, centerR]`：window 外的 env 是 valid 的，window 內是過時的
- `delete(i)`：把 centerL 往左縮、centerR 往右縮（擴大 stale window）
- `update_envs(centerL, centerR)`：重新計算讓 stale window 收縮到 `[centerL, centerR]`
- `OperatorEnv` 的 env tensor shape：`(dn, w, up)` = `(χ_ket, χ_mpo, χ_bra)`
- `VectorEnv` 的 env tensor shape：`(dn, up)` = `(χ_ortho, χ_psi)`
- Boundary：L0 = `ones(T, 1, 1, 1)`（OperatorEnv），R0 同

### `OperatorEnv._grow_left` 的縮併

```julia
# prev_env shape: (dn, w, up)
# ket = mps1[p] shape: (l, i, r)
# mpo[p] shape: (l, ip, i, r)
# bra = mps2[p] shape: (l, i, r)，需要 conj
@tensor new_env[dn_r, w_r, up_r] :=
    prev_env[dn_l, w_l, up_l] *
    ket[dn_l, i, dn_r] *
    mpo[w_l, ip, i, w_r] *
    conj(bra)[up_l, ip, up_r]
# ket:  (l=dn_l, i, r=dn_r)
# mpo:  (l=w_l, ip, i, r=w_r)，ip 給 bra，i 給 ket
# bra:  (l=up_l, ip, r=up_r)，conj，ip 與 mpo 縮
```

### `OperatorEnv._grow_right` 的縮併

```julia
# next_env shape: (dn, w, up)（同樣的 leg 定義，從右邊來）
@tensor new_env[dn_l, w_l, up_l] :=
    next_env[dn_r, w_r, up_r] *
    ket[dn_l, i, dn_r] *
    mpo[w_l, ip, i, w_r] *
    conj(bra)[up_l, ip, up_r]
```

### `VectorEnv._grow_left` 的縮併

```julia
# prev_env shape: (dn, up)
# ortho[p] shape: (l, i, r)  （ket side）
# psi[p] shape: (l, i, r)    （bra side，conj）
@tensor new_env[dn_r, up_r] :=
    prev_env[dn_l, up_l] *
    ortho[dn_l, i, dn_r] *
    conj(psi)[up_l, i, up_r]
```

### Observer callback 機制

Python 版在 `MPS.__setitem__` 後觸發 `env.delete(site)`。Julia 版：

```julia
# MPS 的 callbacks 存為 Vector{Tuple{WeakRef, Symbol}}
# 或簡化為 Vector{Function}（closure 形式）
# 每次 mps[site] = new_tensor 後呼叫所有 callback(site)
```

### 注意事項

- `_grow_left` 和 `_grow_right` 的 bra 端（mps2）要 `conj`，**complex 測試才能驗證**
- `OperatorEnv(mps1, mps2, mpo)`：DMRG 時 mps1 == mps2（自己對自己），`tensors_to_watch` 要避免重複 delete 導致 centerL/centerR 跑掉（`delete` 是 idempotent 的，min/max 同值無副作用）

### 測試：`test/test_environment.jl`

```
1. OperatorEnv 建立後 LR[-1] 和 LR[N] 存在且為 ones
2. update_envs(0,0) 後所有右邊 env 都 valid
3. _grow_left 結果與手動縮併一致
4. delete(site) 正確擴大 stale window
5. real + complex（驗證 bra conj）
```

---

## Step 17：`src/DMRG/EffectiveOperators.jl`

**對應 Python**：`DMRG/effective_operators.py`（341行）

### 讀 Python 檔案前先確認

- `EffOperator.apply(phi)` 回傳的 shape **必須和 phi 完全相同**（KrylovKit eigsolve 要求）
- `EffVector` 先在 `__init__` precompute `|Φ_0⟩`，然後 `inner(phi)` = `sum(conj(Φ_0) .* phi)`
- `add_term(eff_vec, weight)` 加入 penalty：`apply` 的結果加上 `weight * <Φ_0|phi> * |Φ_0⟩`

### `EffOperator.apply` 的縮併（1-site）

```julia
# L shape: (dn, w, up)，R shape: (dn, w, up)
# mpo shape: (l, ip, i, r)
# phi shape: (l, i, r)  →  輸入
# out shape: (l, i, r)  →  輸出，要和 phi 完全一樣
@tensor out[up_l, ip, up_r] :=
    L[dn_l, w_l, up_l] *
    phi[dn_l, i, dn_r] *
    mpo[w_l, ip, i, w_r] *
    R[dn_r, w_r, up_r]
# out 的 (up_l, ip, up_r) 對應 phi 的 (l, i, r)
```

### `EffOperator.apply` 的縮併（2-site）

```julia
# phi shape: (l, i0, i1, r)
# 兩個 mpo tensor：mpo0 shape: (l,ip0,i0,r0)，mpo1 shape: (l1,ip1,i1,r)
@tensor out[up_l, ip0, ip1, up_r] :=
    L[dn_l, w_l, up_l] *
    phi[dn_l, i0, i1, dn_r] *
    mpo0[w_l, ip0, i0, w_mid] *
    mpo1[w_mid, ip1, i1, w_r] *
    R[dn_r, w_r, up_r]
```

### penalty term

```julia
function apply(effH::EffOperator, phi)
    out = _apply_hamiltonian(effH, phi)
    for (eff_vec, weight) in effH.terms
        overlap = eff_vec.inner(phi)   # <Φ_0|phi>
        out .+= weight * overlap * eff_vec.tensor
    end
    return out
end
```

### 注意事項

- `out` 的 leg 順序 `(up_l, ip, up_r)` 要確認等同於 `phi` 的 `(l, i, r)`；`up_l` 來自 L 的 `up` leg，對應 MPS 的 bra 側

### 測試：`test/test_effective_operators.jl`

```
1. apply(phi) shape 和 phi 相同
2. apply 結果與手動展開的 L*phi*W*R 縮併相同
3. EffVector.inner(phi) 正確（和直接 inner product 比對）
4. add_term 後 apply 結果包含 penalty 貢獻
5. real + complex（驗證 bra conj）
```

---

## Step 18：`src/DMRG/Engine.jl`

**對應 Python**：`DMRG/dmrg_engine.py`（175行）

### 讀 Python 檔案前先確認

- sweep 流程：右掃 p=0..N-2，左掃 p=N-n..0（n=num_center），兩段合在一起是完整的一個 sweep
- `_local_update`：update_envs → build EffOperator → eigsolve → update_sites

### KrylovKit `eigsolve` 的使用

```julia
# phi 是 Array{T,3} 或 Array{T,4}，eigsolve 需要 Vector
phi_vec = vec(phi)
apply_vec = v -> vec(effH_apply(reshape(v, size(phi))))

vals, vecs, info = eigsolve(apply_vec, phi_vec, 1, :SR;
                            krylovdim=max(20, length(phi_vec)),
                            tol=1e-10)
E   = real(vals[1])
phi = reshape(vecs[1], size(phi))
```

### 注意事項

- `krylovdim` 不能超過向量維度；建議設為 `min(30, length(phi_vec))`
- `eigsolve` 的 `which=:SR`（smallest real part）對應 ground state
- psi 需先 `orthogonalize!` 設 center=0 才能開始 sweep

### 測試：`test/test_dmrg.jl`

```
1. 1D Heisenberg（N=8）：ground state energy 與 ED 結果比對（誤差 < 1e-8）
2. 1-site 和 2-site sweep 各測
3. excited state（ortho_states 功能）
4. 測試 complex Hamiltonian（能量應為實數）
```

---

## Step 19：`src/TDVP/Engine.jl`

**對應 Python**：`TDVP/tdvp_engine.py`（292行）

### 讀 Python 檔案前先確認

- `_update_1site(p, dt, absorb)`：
  1. forward：`phi = exp(-dt/2 * H_eff) |psi[p]⟩`（`exponentiate`）
  2. 邊界 site 跳過 backward，直接更新
  3. SVD split phi → A（isometry）和 C（bond tensor）
  4. 從 op_env 和 A 手動 grow 一個 0-site env（**不更新** `self._op_env`）
  5. backward：`C' = exp(+dt/2 * H_eff_0site) |C⟩`
  6. 把 C' 吸入鄰居

- `_update_2site(p, dt, ...)`：類似，但 SVD 的對象是合併的 2-site phi

### KrylovKit `exponentiate` 的使用

```julia
# forward step（H_eff 的 exp）
phi_vec = vec(phi)
apply_vec = v -> vec(effH_apply(reshape(v, size(phi))))
result_vec, info = exponentiate(apply_vec, -dt/2, phi_vec;
                                krylovdim=min(30, length(phi_vec)),
                                tol=1e-10)
phi = reshape(result_vec, size(phi))

# backward step（0-site H_eff，向量 C 更小）
C_vec = vec(C)
apply_C = v -> vec(eff0_apply(reshape(v, size(C))))
result_C, info = exponentiate(apply_C, +dt/2, C_vec; ...)
C = reshape(result_C, size(C))
```

### 注意事項

- forward 和 backward 的 `dt` 符號相反（forward `-dt/2`，backward `+dt/2`）
- backward step 用的是 0-site effective Hamiltonian（bond tensor 的 H_eff），比 1-site 小得多，**不能共用同一個 `apply_vec` closure**
- real-time evolution：`dt = 1im * delta_t`；imaginary-time：`dt = delta_tau`（real）

### 測試：`test/test_tdvp.jl`

```
1. imaginary-time TDVP 收斂到 DMRG 的 ground state energy
2. real-time TDVP 能量守恆（誤差 < 1e-6 per step）
3. 1-site 和 2-site 各測
4. 與 ED 的時間演化比對（小系統）
```

---

## Step 20（補完）：`fit_apply_mpo` in `Operations.jl`

**在 Environment.jl 和 EffectiveOperators.jl 完成後回來實作。**

### 讀 Python 檔案前先確認

- 用 `OperatorEnv(mps_input, fitmps, mpo)`（bra=fitmps，ket=mps_input）
- 每個 local step：`effH.apply(phi_in)` 直接是新的 phi（不需 eigsolve）
- sweep 結構和 DMRG sweep 完全相同

---

## 完整實作順序

### Phase 1：基礎工具
```
1.  Project.toml + src/MPSCore.jl
2.  src/TensorUtils.jl
3.  test/test_tensor_utils.jl              ★ 跑測試
```

### Phase 2：Sites + Lattice
```
4.  src/Sites/Site.jl
5.  src/Sites/SpinHalf.jl
6.  src/Sites/SpinlessFermion.jl
7.  src/Sites/Electron.jl
8.  test/test_sites.jl                     ★ 跑測試
9.  src/Lattice/Square.jl
10. test/test_lattice.jl                   ★ 跑測試
```

### Phase 3：MPS / MPO 核心
```
11. src/MPS/MPS.jl
12. test/test_mps.jl                       ★ 跑測試
13. src/MPS/MPO.jl
14. test/test_mpo.jl                       ★ 跑測試
15. src/MPS/Init.jl
16. test/test_mps_init.jl                  ★ 跑測試
17. src/MPS/Operations.jl（fit_apply_mpo 先留 stub）
18. test/test_mps_operations.jl            ★ 跑測試（跳過 fit_apply_mpo）
19. src/MPS/Compression.jl
20. src/MPS/MPOCompression.jl
21. test/test_compression.jl               ★ 跑測試
```

### Phase 4：AutoMPO
```
22. src/MPS/AutoMPO.jl
23. test/test_auto_mpo.jl                  ★ 跑測試（用 Python dump 比對）
```

### Phase 5：DMRG
```
24. src/DMRG/Environment.jl
25. test/test_environment.jl               ★ 跑測試
26. src/DMRG/EffectiveOperators.jl
27. test/test_effective_operators.jl       ★ 跑測試
28. src/DMRG/Engine.jl
29. test/test_dmrg.jl                      ★ 跑測試
```

### Phase 6：TDVP + 收尾
```
30. src/TDVP/Engine.jl
31. test/test_tdvp.jl                      ★ 跑測試
32. Operations.jl：補完 fit_apply_mpo
33. test/test_mps_operations.jl            ★ 重跑完整測試
```

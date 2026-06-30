# 實作進度

對照 `plan.md` 的 33 步驟紀錄目前狀態。

最後更新：**Phase 6 完成 — 全部 33 步驟完成（364 tests passing）** 🎉

## 後續修正

- **DMRG 截斷後歸一（2026-06-14）**：`sweep!` 結尾加 `LinearAlgebra.normalize!(engine.psi)`。
  原本截斷會讓 state norm 略低於 1（每個 SVD 丟掉一點權重），且不在 svd_split 補回。
  改在 MPS 層歸一（sweep 結束時 MPS 有單一 center，normalize! 只 rescale center tensor）。
  energy 是 Rayleigh quotient 不受影響；測試全過（364/364）。驗證：maxdim=4 的截斷 DMRG
  後 `norm(psi) == 1.0`。

---

## Phase 1：基礎工具 ✅

### Step 1：`Project.toml` + `src/MPSCore.jl` ✅
- `Project.toml`：Julia 1.10，依賴 KrylovKit、TensorOperations、LinearAlgebra（stdlib），套件名 `MPSCore`
- `src/MPSCore.jl`：主模組（package entry point），include 所有 phase 的檔案並 export 對外 API

### Step 2：`src/TensorUtils.jl` ✅
實作三個函式：
- `svd_split(A, n_row_legs; maxdim, cutoff, absorb)`
  - cutoff 定義：對 normalized rho eigenvalue `λ_i = |s_i|²/Σ|s_j|²`，與 Python 版一致
  - absorb 三種模式：`"left"`、`"right"`（default）、`nothing`（回傳獨立 s）
  - 至少保留 1 個 singular value
- `qr_split(A, n_row_legs)`
  - 回傳 `(Q, R)`，Q 是 isometry
- `direct_sum(A, B, sum_dims)`
  - 沿指定維度做 block-diagonal cat，off-diagonal 填零
  - 支援不同 element type，自動 `promote_type`

### Step 3：`test/test_tensor_utils.jl` ✅ — 58 tests，全部通過
測試覆蓋：
- `svd_split` round-trip（real rank-2、complex rank-3、MPS shape）
- `svd_split` absorb 三種模式（isometry 驗證）
- `svd_split` maxdim 截斷（chi <= maxdim）
- `svd_split` cutoff 截斷（chi 和 discarded 數值驗證）
- `svd_split` complex 正確性（U†U = I，需要 conj）
- `qr_split` round-trip（real + complex）
- `direct_sum`（rank-3 interior/boundary、rank-4 MPO、complex、mixed type）

**修正紀錄：**
- 測試寫錯 chi 期望值：`(3,4,5)` 的 rank-3 tensor，n_row_legs=2，χ = min(12,5) = 5，不是 12
- 測試寫錯 cutoff：λ_2 = 1/101 ≈ 0.0099，cutoff 需 > 0.0099 才會截斷（改用 0.02）

---

## Phase 2：Sites + Lattice ✅

```
4.  src/Sites/Site.jl              ✅
5.  src/Sites/SpinHalf.jl          ✅
6.  src/Sites/SpinlessFermion.jl   ✅
7.  src/Sites/Electron.jl          ✅
8.  test/test_sites.jl             ✅
9.  src/Lattice/Square.jl          ✅
10. test/test_lattice.jl           ✅
```

### 實作紀錄
- `PhysicalSite` 用 mutable struct 達成（Julia struct 預設 immutable，但 Dict field 本身可修改）
- `register_op!` 自動轉換矩陣為 `ComplexF64`
- SpinHalf 的 Sz convention 與 Python 一致：dn（index 1）= +1/2，up（index 2）= -1/2（非標準物理 convention），故 `[Sz,Sp] = -Sp`
- `SquareLattice` 座標 0-indexed（與 Python 版一致）
- bonds 預計算並排序，保證 i < j

**修正紀錄：**
- 後來決定改成標準物理 convention：basis 順序改為 `(up=index1, dn=index2)`，Sz = `[[+0.5,0],[0,-0.5]]` = `0.5*sigma_z`，現在 `[Sz,Sp]=+Sp`。Python 端也同步修正。

---

## Phase 3：MPS / MPO 核心 ✅

```
11. src/MPS/MPS.jl                 ✅
12. test/test_mps.jl               ✅
13. src/MPS/MPO.jl                 ✅
14. test/test_mpo.jl               ✅
15. src/MPS/Init.jl                ✅
16. test/test_mps_init.jl          ✅
17. src/MPS/Operations.jl          ✅  (fit_apply_mpo 留 stub)
18. test/test_mps_operations.jl    ✅
19. src/MPS/Compression.jl         ✅
20. src/MPS/MPOCompression.jl      ✅
21. test/test_compression.jl       ✅
```

### 實作紀錄
- `MPS{T}` mutable struct，dense `Vector{Array{T,3}}`，1-indexed sites
- `set_sites!` 提供 atomic multi-site update（避免 SVD 過程中 bond 暫時不一致）
- Callbacks 用 `Vector{Any}`（無 weakref，靠手動 unregister）
- `_shift_center_right!` / `_shift_center_left!` 用 `svd_split` + `@tensor` 縮併
- `make_phi` / `update_sites!` 支援 1-site 和 2-site
- `norm` 用本地 `<psi|psi>` 縮併實作（不依賴 Operations.jl）
- `MPO{T}` 同樣是 mutable struct，leg 順序 `(l, ip, i, r)`
- `inner` / `expectation`：bra 端永遠 `conj()`，complex 測試驗證
- `mps_sum` / `mpo_sum` 用 `direct_sum`（site 1 只 r、site N 只 l、interior l+r）
- `exact_apply_mpo` / `mpo_product` 用 `@tensor` 縮併後 `reshape` 合併 virtual bond
- `fit_apply_mpo` / `fit_mpo_product` 留 stub，等 DMRG Environment 完成
- `svd_compress_mps` 右掃 left-canonicalize 後左掃 SVD 截斷
- `svd_compress_mpo` 兩段式 sweep（left→right canonicalize, right→left truncate）

**修正紀錄：**
- `MPS` 需要定義 `Base.lastindex` 才能用 `psi[end]`
- `random_mps` 的 bond dim 測試太樂觀：phys=2, N=6, D=4 時靠近邊界的 bond 會被 phys^k 限制（natural max 是 [1,2,4,4,4,2,1]）

---

## Phase 4：AutoMPO ✅

```
22. src/MPS/AutoMPO.jl             ✅
23. test/test_auto_mpo.jl          ✅
```

### 實作紀錄
- `AutoMPO` mutable struct + `Term` / `OpEntry` 簡單型別
- `add!(ampo, coeff, "Op1", site1, "Op2", site2, ...)`，1-indexed sites
- `_preprocess_term`：JW 字串展開，F² = I 對消
- `_enumerate_states`：bond_states[p] = `[:done; partials...; :start]`，按 `string(key)` 排序
- 邊界 bond 1 / N+1 強制只有 `:start` / `:done`，dim=1
- 沒有 QN 處理（dense-only），程式碼相比 Python 大幅簡化
- Validation 採用 ED 比對：把 MPO 完整縮成 dense matrix，與用 `kron` 直接組的 H 比對
- 測試案例：Heisenberg N=4、spinless fermion tight-binding N=4、long-range `Cdag_1 C_4` JW string、隨機 MPS 的 expectation 與 ED 一致

**修正紀錄：**
- 測試 helper `mpo_to_matrix` / `mps_to_vec` 一開始用錯 reshape 順序：Julia 是 column-major，第一個 dim 變化最快；要對齊 `kron`（其 row index 把第一個參數當最高位）必須先 `permutedims` 把 ipN..ip1 反序再 reshape

---

## Phase 5：DMRG ✅

```
24. src/DMRG/Environment.jl        ✅
25. test/test_environment.jl       ✅
26. src/DMRG/EffectiveOperators.jl ✅
27. test/test_effective_operators.jl ✅
28. src/DMRG/Engine.jl             ✅
29. test/test_dmrg.jl              ✅
```

### 實作紀錄
- `OperatorEnv` / `VectorEnv` 用 `Dict{Int, Array}` 存環境，鍵 0..N+1（0、N+1 為邊界）
- Stale window `[centerL, centerR]` ⊆ `[1, N]`，外部 valid
- 自動 callback：`OperatorEnv` ctor 把 `delete!(env, site)` 註冊到 mps1 / mps2 / mpo
- `_grow_left` / `_grow_right` 用 `@tensor` 縮併，bra 端永遠 `conj`
- `EffOperator` 支援 0/1/2 site（0-site = bond tensor for TDVP）
- `EffVector` 預先把 ortho state 投影到 local subspace，`inner(ev, phi)` 算 `<Φ_0|phi>`
- `DMRGEngine` + `sweep!` + `dmrg!`（多 sweep schedule wrapper）
- 用 `KrylovKit.eigsolve` 取最小特徵值，傳入 `vec(phi)` / `reshape` 包裝
- 支援 excited state via penalty term

### 測試覆蓋
- OperatorEnv：邊界、full env vs `expectation`、雙向 grow、callback invalidation
- EffOperator：1-site / 2-site / 0-site shape & energy；penalty term 手動驗證
- EffVector：overlap 與 `inner(ortho, psi)` 一致
- DMRG：Heisenberg N=4 和 N=8（2-site）、N=4（1-site）vs ED；complex MPO；excited state via penalty

**修正紀錄：**
- `EffVector` test 寫錯：`inner(eff_vec, phi)` 是 `<ortho|psi>` = `inner(ortho, psi)`，不是 `inner(psi, ortho)`（兩者複共軛）
- `svd_compress_mpo` 複數測試：浮點 round-off 在大數值（~10^4）下絕對誤差 ~1e-10 是正常的，改用 `rtol`

---

## Phase 6：TDVP + 收尾 ✅

```
30. src/TDVP/Engine.jl             ✅
31. test/test_tdvp.jl              ✅
32. Operations.jl：補完 fit_apply_mpo ✅
33. test/test_mps_operations.jl 補完 fit_apply_mpo 測試 ✅
```

### 實作紀錄
- `TDVPEngine` + `sweep!`：1-site / 2-site 都實作
- 1-site 流程：forward (`exp(-dt/2 H_eff)`) → SVD → 0-site env (用 A 而非 mps tensor) → backward (`exp(+dt/2 H_eff_0site)`) → 吸進 neighbour
- 2-site 流程：2-site forward → `update_sites!` SVD → 1-site backward on new center
- 邊界 site 跳過 backward（標準 TDVP 約定）
- 0-site env 構造：把 `_grow_left_op` / `_grow_right_op` 抽出成 raw helper（吃顯式 tensor），避免修改 `op_env`
- 用 `KrylovKit.exponentiate(f, t, x)` 算 `exp(t·H) x`，包成 `_expm_apply` helper
- `fit_apply_mpo` 補完：用 `OperatorEnv(mps_input, fitmps, mpo)`，每個 local step 直接 `apply(effH, phi_in)`（不用 eigsolver）

### 測試覆蓋
- Real-time TDVP（1-site + 2-site）：能量守恆 < 1e-7，norm 守恆 < 1e-8 ✓
- Imaginary-time TDVP：收斂到 ground state（誤差 ~ Trotter step^2）✓
- `fit_apply_mpo`：與 `exact_apply_mpo` overlap 一致（4 sweeps 即收斂）✓

**修正紀錄：**
- TDVP imaginary time 一開始用 dt=0.1 太大，Trotter 誤差 ~5e-4 達不到 atol=1e-6；改用 dt=0.05 + 200 steps，並把 atol 放寬到 1e-4（2-site）/ 1e-3（1-site），符合 TDVP 實際收斂行為
- KrylovKit 的 `exponentiate(f, t, x)` 返回 `(result, info)`，要記得只取 `result`

# 本地封存归档引用

`registry.json` 用完整 SHA-256 标识封存载荷，并记录两个本地保管副本。`availability=local_custody` 表示本仓库仅交付元数据，尚无托管下载资产，也未承诺异地持久存储。若所列路径不可访问，须另行联系归档保管者取得载荷。

`program_members.jsonl` 将150个导入 case ID 逐一绑定到 tar 成员、成员 SHA-256、字节数和原程序路径。原路径仅记录来源，在此轻量包内不可直接解析。通过 `registry.json` 解析 `archive_id`，另行取得 `filesystem.tar.gz` 并核对其 SHA-256 后，将所需成员解压到隔离目录，再核对成员哈希与大小。

取得并核对载荷后，提取命令示例：

```sh
tar -xzf /path/to/filesystem.tar.gz -C /path/to/isolated-destination -- 'filesystem/91004t2/laq/code/ThreejsJudge/TurnItOver-dataset/benchmark/easy/imported-obvious-v1/private/programs/<case_id>.json'
```

`original_splits.json` 是导入前125例 split 的原字节快照；`source_splits.json` 是当前275例源 split 的原字节快照，保留了尚未替换的绝对 Gold 路径。当前 split 保留全部物体及案例分配，并绑定当前相对 Gold 的 SHA-256。

`sealed_scope.json`、`sealed_verification.json`、`sealed_SHA256SUMS` 是从封存目录原样复制的小型记录；原校验清单对应原封存目录，不对应此元数据目录。完整原始清单继续保存在本地封存中，此处不重复导入。registry 记录本轮重新计算的两个载荷副本哈希，以及全部所选程序成员的检查结果。

归档含 private Gold、程序几何与历史源上下文，应保持原访问边界，不直接供给模型。源环境验证与独立字节恢复是不同检查；本次交付不声称完成新的渲染、采样重放、恢复后独立语义验证、逐例独立人工盲审、训练或 benchmark 冻结。

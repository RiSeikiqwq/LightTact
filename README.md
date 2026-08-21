# LightTact DContact — 还原版软件

根据 LightTact 论文（*LightTact: A Visual–Tactile Fingertip Sensor for
Deformation-Independent Contact Sensing*，Sec. III-D Contact Segmentation）
与作者团队的半成品 `calibration_2.py` 还原的相机标定 + 接触分割软件。
算法实现与论文逐条对应；作者缺失的 `camera_2.py` 相机模块与
`shape_config.yaml` 配置已补齐（参考 9DTact 开源仓库的路径/校准模式）。

## 论文要点 → 代码对照

| 论文要点 (III-D) | 代码位置 |
|---|---|
| 固定 20 ms 曝光、关闭自动曝光 | `camera_2.py` `Camera._apply_exposure`（V4L2 用 100µs 单位 200；MSMF/DSHOW 用 ms；读回验证；不支持时仅警告） |
| 5×5 圆柱凸点标定板（3 mm 间距）→ 阈值分割 → 中心点 → 校正网格 | `calibration_2.py` `run_calibration`（作者原算法，逐字节未动） |
| N=10 帧参考均值 I_ref | `camera_2.py` `get_raw_avg_image`（`reference_frames: 10`） |
| 四条件亮度一致性测试 t0,t1,t2,t3=25,20,30,40 | `camera_2.py` `segment_contact`（全向量化） |
| 原始↔校正映射存 npy，下次复用免重标定 | `row_index.npy` / `col_index.npy` / `position_scale.npy` / `valid_mask_cropped.npy`，`Camera(calibrated=True)` 自动加载 |

## 文件结构

```
lighttact_dcontact/
├── camera_2.py              # Camera 类（捕获/曝光/分割/校正）+ segment_contact + make_camera
├── calibration_2.py         # 作者原标定算法（仅补 headless/曝光/Unicode 写图等最小改动）
├── _1_Camera_Calibration.py # 标定入口：python _1_Camera_Calibration.py
├── dcontact.py              # DContact 运行时 API（机器人集成入口）
├── _2_DContact_Demo.py      # 实时演示：raw/分割/校正窗口
├── mock_camera.py           # 模拟相机（无硬件时开发/测试用）
├── test_pipeline.py         # 无头端到端测试（45 项断言）
├── shape_config.yaml        # 配置（曝光、ROI、阈值、路径）
├── requirements.txt
└── test_output/             # 测试生成的产物（可随时删除）
```

## 安装

```bash
cd lighttact_dcontact
python3 -m venv --copies .venv        # E 盘(drvfs)上需要 --copies
.venv/bin/python3 -m pip install -r requirements.txt
```
> 若 `ensurepip` 不可用（部分精简版 Python），用 `curl -sSL
> https://bootstrap.pypa.io/get-pip.py | .venv/bin/python3 -` 补装 pip。
> 系统 Python 已装有 numpy/opencv/scipy/PyYAML 时也可直接 `python3 xxx.py`。

## 真机使用（OV5693 UVC 相机）

```bash
.venv/bin/python3 _1_Camera_Calibration.py
```
1. 不要触碰传感器，按 `y` 采集参考图（10 帧平均）；
2. 将 5×5 标定板压在传感面上，按 `y` 采集样本图；
   预览窗口中 `+`/`-` 可实时调整曝光时间（论文：固定 20 ms），`e` 恢复配置值；
3. 脚本自动检测 25 个压印点、计算校正映射并保存 npy（校准完成，之后可直接复用）；
4. 调试图保存在 `calibration/sensor_<id>/camera_calibration/`。

```bash
.venv/bin/python3 _2_DContact_Demo.py
```
实时分割演示：`+`/`-` 调曝光，`y` 重采参考，`r` 保存调试图，`q` 退出。

## 无硬件（模拟器）

```bash
.venv/bin/python3 _1_Camera_Calibration.py --camera mock --headless
.venv/bin/python3 _2_DContact_Demo.py    --camera mock --headless --frames 20
.venv/bin/python3 test_pipeline.py       # 45 项断言，全部 PASS
```

## 机器人集成（DContact API）

```python
from dcontact import DContact

dc = DContact(backend="real")          # 自动加载已保存的标定映射与参考图
if dc.in_contact():                    # 接触比例 > min_contact_ratio (0.001)
    mask, img_rect = dc.detect_rectified()   # 校正后 (H_out, W_out) 视图，各向同性 px/mm
    ratio = dc.contact_ratio(mask)     # 接触面积占比
    dc.capture_reference()             # 需要时重新采集参考图
```

## 输出文件（标定后）

| 文件 | 内容 |
|---|---|
| `position_scale.npy` | `[中心行, 中心列, mm/像素(行), mm/像素(列)]`（裁剪坐标） |
| `row_index.npy` / `col_index.npy` | 校正空间→裁剪原始空间 的密集映射（int32，`img[row_index, col_index]` 即校正图） |
| `valid_mask_cropped.npy` | 传感有效区凸包掩码（校正后用于剔除感应区外像素） |
| `ref.npy` | 参考图（DContact 采集后自动保存，下次启动自动加载） |
| 各 PNG | 参考/样本/接触掩码/梯形调试图 |

注：`calibration/` 目录为真机标定产物；`test_output/` 为测试产物（可删）。

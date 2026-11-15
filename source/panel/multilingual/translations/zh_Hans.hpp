// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

// This file is also dual licensed under the Apache License, Version 2.0. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>

#pragma once

#include <array>

namespace zlpanel::multilingual::zh_Hans {
    inline constexpr std::array kTexts = {
        // band basic controls
        "切换频段旁通状态。",
        "切换频段独奏监听。",
        "选择滤波器类型。",
        "选择滤波器斜率。斜率越高，滤波器的频率响应曲线变化越陡峭。",
        "选择立体声通道模式。",
        "调节中心频率或截止频率。",
        "调节基础增益与动态目标增益。",
        "调节品质因数 (Q 值)。Q 值越大，带宽越窄。",
        "切换此频段的动态均衡行为。",
        "关闭此频段。",
        "选择上一个频段。",
        "选择下一个频段。",
        "切换低通和高通滤波器的响应缩放。",
        "调节低通或高通滤波器的幅度缩放比例。",
        // band dynamic controls
        "切换此频段的动态处理旁通状态。",
        "切换频谱动态差分信号的输出。",
        "选择频谱动态模式。",
        "动态处理使用静态阈值。",
        "动态处理使用与该频段侧链响度相关的动态阈值。",
        "动态处理使用与总侧链响度相关的动态阈值。",
        "调节静态阈值。",
        "调节频段相对阈值。",
        "调节总相对阈值。",
        "调节动态处理的拐点宽度。",
        "调节动态处理的启动时间。",
        "调节动态处理的释放时间。",
        // top controls
        "切换界面设置面板。",
        "切换预设浏览器。",
        "切换分析仪设置面板。",
        "切换输出设置面板。",
        "切换插件旁通状态。",
        "切换外部侧链输入。",
        // preset browser
        "在系统文件管理器中打开预设文件夹。",
        "删除所选预设分组。",
        "删除所选预设。",
        "关闭预设浏览器。",
        // spectrum controls
        "切换频谱处理设置面板。",
        "选择频谱处理分辨率。",
        "选择频谱处理侧链的频率平滑模式。",
        "调节频谱处理侧链的频率平滑量。",
        "调节频谱处理侧链倾斜斜率。",
        "调节频谱处理侧链启动偏置。偏置值越高，高频部分的启动越快。",
        "调节频谱处理侧链释放偏置。偏置值越高，高频部分的释放越快。",
        "调节相对响度的频谱门限。",
        // output controls
        "调节额外的输出增益。",
        "调节所有滤波器基础增益与目标增益的整体缩放比例。",
        "切换静态增益补偿 (SGC)。SGC 为近似补偿，但不会影响动态。",
        "切换响度增益匹配。开启时测量输入与输出响度，关闭时将输出增益设为两者的响度差值。",
        // analyzer controls
        "切换输入信号频谱的显示。",
        "切换输出信号频谱的显示。",
        "切换侧链信号频谱的显示。",
        "选择频谱分析仪的衰减速度。",
        "选择频谱分析仪的倾斜斜率。",
        "选择频谱分析仪的频率平滑宽度。",
        "选择频谱分析仪的频率平滑模式。\nOCT：等比例分数倍频程平滑。\nERB：符合人耳听觉的等效矩形带宽平滑。",
        "切换频谱冻结功能。在分析仪上保持鼠标静止时冻结频谱。开启后，也可以通过 Shift 键切换冻结。",
        "选择频谱分析仪显示的立体声通道模式。",
        "切换频率碰撞检测。",
        "调节碰撞检测的强度。",
        "切换数值读数的显示。",
        "切换电平表栏的显示。",
        // curve scale & piano roll
        "选择均衡响应曲线的分贝显示范围。",
        "选择频谱分析仪显示的上限分贝值。",
        "选择频谱分析仪显示的最小分贝下限。",
        "切换钢琴卷帘的显示。",
        " "
    };
}

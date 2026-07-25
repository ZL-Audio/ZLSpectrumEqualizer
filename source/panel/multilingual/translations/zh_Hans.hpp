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
    static constexpr std::array kTexts = {
        "释放：旁路该频段。",
        "按下：独奏该频段。",
        "选择滤波器类型。",
        "选择滤波器斜率。较高的斜率会让滤波器的响应曲线变化更陡峭。",
        "选择声道模式。",
        "调节频率。",
        "调节基础增益与目标增益。",
        "调节品质因数。更大的 Q 值会让带宽更窄。",
        "按下：开启动态行为。",
        "点击：关闭该频段。",

        "释放：旁路动态处理。",
        "选择动态模式。",
        "动态处理使用静态阈值。",
        "动态处理使用与该频段侧链响度相关的动态阈值。",
        "动态处理使用与总侧链响度相关的动态阈值。",
        "调节静态阈值。",
        "调节频段相对阈值。",
        "调节总相对阈值。",
        "调节动态处理的拐点宽度。",
        "调节动态处理的启动时间。",
        "调节动态处理的释放时间。",

        "释放：旁路插件。",
        "按下：使用外部侧链。\n释放：使用内部侧链。",

        "选择频谱处理分辨率。",
        "选择频谱处理侧链平滑类型。",
        "选择频谱处理侧链平滑值。",
        "调节频谱处理侧链倾斜斜率。",
        "调节频谱处理侧链启动偏置。偏置值越高，高频部分的启动越快。",
        "调节频谱处理侧链释放偏置。偏置值越高，高频部分的释放越快。",
        "调节相对响度的频谱门限。",

        "调节额外输出增益。",
        "调节所有滤波器的基础增益与目标增益缩放比例。",
        "按下：开启静态增益补偿（SGC）。SGC 不够精确，但不会影响动态。",
        "按下：开始测量输入信号与输出信号的整体响度\n释放：将输出增益更新为两响度值之差。",

        "按下：开启输入信号频谱分析仪",
        "按下：开启输出信号频谱分析仪",
        "按下：开启侧链信号频谱分析仪",
        "选择频谱分析仪的衰减速度。",
        "选择频谱分析仪的倾斜斜率。",
        "选择频谱分析仪的平滑值。",
        "选择频谱分析仪的平滑类型。",
        "按下：开启冻结功能。将鼠标悬停在分析仪上即可冻结频谱。",
        "按下：开启碰撞检测。",
        "调节碰撞检测强度。",

        "双击：打开界面设置。",
    };
}

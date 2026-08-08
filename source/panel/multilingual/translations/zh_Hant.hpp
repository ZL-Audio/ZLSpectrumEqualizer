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

namespace zlpanel::multilingual::zh_Hant {
    static constexpr std::array kTexts = {
        "釋放：旁路該頻段。",
        "按下：獨奏該頻段。",
        "選擇濾波器類型。",
        "選擇濾波器斜率。較高的斜率會讓濾波器的響應曲線變化更陡峭。",
        "選擇聲道模式。",
        "調整頻率。",
        "調整基礎增益與目標增益。",
        "調整品質因數。更大的 Q 值會讓頻寬更窄。",
        "按下：開啟頻譜動態行為。",
        "點擊：關閉該頻段。",

        "釋放：旁路頻譜動態處理。",
        "按下：輸出頻譜動態差值訊號。",
        "選擇頻譜動態模式。",
        "動態處理使用靜態閾值。",
        "動態處理使用與該頻段側鏈響度相關的動態閾值。",
        "動態處理使用與總側鏈響度相關的動態閾值。",
        "調整靜態閾值。",
        "調整頻段相對閾值。",
        "調整總相對閾值。",
        "調整動態處理的拐點寬度。",
        "調整動態處理的啟動時間。",
        "調整動態處理的釋放時間。",

        "釋放：旁路外掛程式。",
        "按下：使用外部側鏈。\n釋放：使用內部側鏈。",

        "選擇頻譜處理解析度。",
        "選擇頻譜處理側鏈平滑類型。",
        "選擇頻譜處理側鏈平滑值。",
        "調整頻譜處理側鏈傾斜斜率。",
        "調整頻譜處理側鏈啟動偏置。偏置值越高，高頻部分的啟動越快。",
        "調整頻譜處理側鏈釋放偏置。偏置值越高，高頻部分的釋放越快。",
        "調整相對響度的頻譜門限。",

        "調整額外輸出增益。",
        "調整所有濾波器的基礎增益與目標增益縮放比例。",
        "按下：開啟靜態增益補償（SGC）。SGC 不夠精確，但不會影響動態。",
        "按下：開始測量輸入訊號與輸出訊號的整體響度\n釋放：將輸出增益更新為兩響度值之差。",

        "按下：開啟輸入訊號頻譜分析儀",
        "按下：開啟輸出訊號頻譜分析儀",
        "按下：開啟側鏈訊號頻譜分析儀",
        "選擇頻譜分析儀的衰減速度。",
        "選擇頻譜分析儀的傾斜斜率。",
        "選擇頻譜分析儀的平滑值。",
        "選擇頻譜分析儀的平滑類型。",
        "按下：開啟凍結功能。將滑鼠懸停在分析儀上即可凍結頻譜。",
        "按下：開啟碰撞檢測。",
        "調整碰撞檢測強度。",

        "雙擊：開啟介面設定。",
    };
}

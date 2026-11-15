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
    inline constexpr std::array kTexts = {
        // band basic controls
        "切換頻段旁通狀態。",
        "切換頻段獨奏監聽。",
        "選擇濾波器類型。",
        "選擇濾波器斜率。斜率越高，濾波器的頻率響應曲線變化越陡峭。",
        "選擇立體聲通道模式。",
        "調節中心頻率或截止頻率。",
        "調節基礎增益與動態目標增益。",
        "調節品質因數 (Q 值)。Q 值越大，頻寬越窄。",
        "切換此頻段的動態均衡行為。",
        "關閉此頻段。",
        "選擇上一個頻段。",
        "選擇下一個頻段。",
        "切換低通和高通濾波器的響應縮放。",
        "調節低通或高通濾波器的幅度縮放比例。",
        // band dynamic controls
        "切換此頻段的動態處理旁通狀態。",
        "切換頻譜動態差分訊號的輸出。",
        "選擇頻譜動態模式。",
        "動態處理使用靜態閾值。",
        "動態處理使用與該頻段側鏈響度相關的動態閾值。",
        "動態處理使用與總側鏈響度相關的動態閾值。",
        "調節靜態閾值。",
        "調節頻段相對閾值。",
        "調節總相對閾值。",
        "調節動態處理的拐點寬度。",
        "調節動態處理的啟動時間。",
        "調節動態處理的釋放時間。",
        // top controls
        "切換介面設定面板。",
        "切換預設瀏覽器。",
        "切換分析儀設定面板。",
        "切換輸出設定面板。",
        "切換插件旁通狀態。",
        "切換外部側鏈輸入。",
        // preset browser
        "在系統檔案管理器中開啟預設資料夾。",
        "刪除所選預設分組。",
        "刪除所選預設。",
        "關閉預設瀏覽器。",
        // spectrum controls
        "切換頻譜處理設定面板。",
        "選擇頻譜處理解析度。",
        "選擇頻譜處理側鏈的頻率平滑模式。",
        "調節頻譜處理側鏈的頻率平滑量。",
        "調節頻譜處理側鏈傾斜斜率。",
        "調節頻譜處理側鏈啟動偏置。偏置值越高，高頻部分的啟動越快。",
        "調節頻譜處理側鏈釋放偏置。偏置值越高，高頻部分的釋放越快。",
        "調節相對響度的頻譜門限。",
        // output controls
        "調節額外的輸出增益。",
        "調節所有濾波器基礎增益與目標增益的整體縮放比例。",
        "切換靜態增益補償 (SGC)。SGC 為近似補償，但不會影響動態。",
        "切換響度增益匹配。開啟時測量輸入與輸出響度，關閉時將輸出增益設為兩者的響度差值。",
        // analyzer controls
        "切換輸入訊號頻譜的顯示。",
        "切換輸出訊號頻譜的顯示。",
        "切換側鏈訊號頻譜的顯示。",
        "選擇頻譜分析儀的衰減速度。",
        "選擇頻譜分析儀的傾斜斜率。",
        "選擇頻譜分析儀的頻率平滑寬度。",
        "選擇頻譜分析儀的頻率平滑模式。\nOCT：等比例分數倍頻程平滑。\nERB：符合人耳聽覺的等效矩形頻寬平滑。",
        "切換頻譜凍結功能。在分析儀上保持滑鼠靜止時凍結頻譜。開啟後，也可以透過 Shift 鍵切換凍結。",
        "選擇頻譜分析儀顯示的立體聲通道模式。",
        "切換頻率碰撞偵測。",
        "調節碰撞偵測的強度。",
        "切換數值讀數的顯示。",
        "切換電平表欄的顯示。",
        // curve scale & piano roll
        "選擇均衡響應曲線的分貝顯示範圍。",
        "選擇頻譜分析儀顯示的上限分貝值。",
        "選擇頻譜分析儀顯示的最小分貝下限。",
        "切換鋼琴捲簾的顯示。",
        " "
    };
}

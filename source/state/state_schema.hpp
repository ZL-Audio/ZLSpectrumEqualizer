// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

namespace zlstate::schema {
    inline constexpr auto kProcessorState = "ParaState";
    inline constexpr auto kParameterState = "Para";
    inline constexpr auto kNonAutomatableState = "State";
    inline constexpr auto kUISettings = "UISetting";

    namespace legacy {
        inline constexpr auto kProcessorState = "ZLSpectrumEqualizerParaState";
        inline constexpr auto kParameterState = "ZLSpectrumEqualizerParameters";
        inline constexpr auto kNonAutomatableState = "ZLSpectrumEqualizerNAParameters";
        inline constexpr auto kUISettings = "ZLSpectrumEqualizerState";
    }
}

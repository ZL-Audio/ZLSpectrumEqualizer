// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <juce_graphics/juce_graphics.h>

namespace zlpanel::preset_style {
    static constexpr auto kTextScale = 1.5f;
    static constexpr auto kBackgroundAlpha = .9f;
    static constexpr auto kSurfaceTint = .05f;

    inline float textFontSize(const float base_font_size) {
        return kTextScale * base_font_size;
    }

    inline juce::Colour surfaceColour(const juce::Colour background, const juce::Colour text) {
        return background.interpolatedWith(text, kSurfaceTint).withAlpha(kBackgroundAlpha);
    }
}

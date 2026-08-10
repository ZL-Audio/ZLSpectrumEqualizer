// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "preset_browser_background.hpp"
#include "preset_style.hpp"
#include "../helper/helper.hpp"

namespace zlpanel {
    PresetBrowserBackground::PresetBrowserBackground(zlgui::UIBase& base) : base_(base) {
        setInterceptsMouseClicks(false, false);
        setAlpha(preset_style::kBackgroundAlpha);
    }

    void PresetBrowserBackground::paint(juce::Graphics& g) {
        const auto font_size = base_.getFontSize();
        const auto padding = getPaddingSize(font_size);
        const auto bound = getLocalBounds().reduced(padding);
        juce::Path path;
        path.addRoundedRectangle(bound.toFloat(), static_cast<float>(padding));

        const juce::DropShadow shadow{base_.getTextColour().withAlpha(.5f), padding, {0, 0}};
        shadow.drawForPath(g, path);
        g.setColour(preset_style::surfaceColour(base_.getBackgroundColour(), base_.getTextColour()));
        g.fillPath(path);

        g.setColour(base_.getBackgroundColour());
        for (const auto& surface_bound : surface_bounds_) {
            g.fillRoundedRectangle(surface_bound.toFloat(), static_cast<float>(padding) * 0.75f);
        }
    }

    void PresetBrowserBackground::setSurfaceBounds(std::vector<juce::Rectangle<int>> bounds) {
        surface_bounds_ = std::move(bounds);
        repaint();
    }
}

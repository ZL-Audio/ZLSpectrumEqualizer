// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include "../../../PluginProcessor.h"
#include "../../../gui/gui.hpp"

namespace zlpanel {
    class SoloPanel final : public juce::Component,
    private juce::ValueTree::Listener {
    public:
        explicit SoloPanel(PluginProcessor& p, zlgui::UIBase &base);

        ~SoloPanel() override;

        void paint(juce::Graphics& g) override;

        bool isSoloSide() const;

        void updateX(float x_left, float x_right);

        void updateBand() const;

    private:
        PluginProcessor& p_ref_;
        zlgui::UIBase& base_;
        zlgui::button::ClickButton solo_whole_button_;
        std::atomic<bool> to_update_{true};
        bool c_solo_side_{false};
        float x_left_{0.f}, x_right_{1e15f};

        void valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier&) override;
    };
}

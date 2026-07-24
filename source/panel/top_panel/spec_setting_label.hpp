// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include "../../PluginProcessor.h"
#include "../../gui/gui.hpp"
#include "../helper/helper.hpp"
#include "../multilingual/tooltip_helper.hpp"

#include "../control_panel/control_background.hpp"

namespace zlpanel {
    class SpecSettingLabel final : public juce::Component {
    public:
        explicit SpecSettingLabel(PluginProcessor& p, zlgui::UIBase& base);

        ~SpecSettingLabel() override;

        void resized() override;

        void repaintCallbackSlow();

    private:
        zlgui::UIBase& base_;
        zlgui::attachment::ComponentUpdater updater_;

        zlgui::combobox::CompactCombobox resolution_box_;
        zlgui::attachment::ComboBoxAttachment<true> resolution_attach_;

        zlgui::combobox::CompactCombobox smooth_type_box_;
        zlgui::attachment::ComboBoxAttachment<true> smooth_type_attach_;
        zlgui::slider::CompactLinearSlider<false, false, false> smooth_slider_;
        zlgui::attachment::SliderAttachment<true> smooth_attach_;

        bool is_over_{false};

        void mouseDown(const juce::MouseEvent&) override;

        void mouseEnter(const juce::MouseEvent&) override;

        void mouseExit(const juce::MouseEvent&) override;

        void updateAlpha(bool is_panel_open);
    };
}

// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include "../../../PluginProcessor.hpp"
#include "../../../gui/gui.hpp"
#include "../../helper/helper.hpp"
#include "../../multilingual/tooltip_helper.hpp"

#include "../../background/panel_background.hpp"

namespace zlpanel {
    class SpecSettingPanel final : public juce::Component {
    public:
        explicit SpecSettingPanel(PluginProcessor& p, zlgui::UIBase& base,
                                  const multilingual::TooltipHelper& tooltip_helper);

        int getIdealWidth() const;

        int getIdealHeight() const;

        void resized() override;

        void repaintCallBackSlow();

    private:
        zlgui::UIBase& base_;
        zlgui::attachment::ComponentUpdater updater_;
        std::atomic<float>& spec_setting_open_;

        PanelBackground control_background_;

        zlgui::label::NameLookAndFeel label_laf_;

        juce::Label tilt_label_;
        zlgui::slider::CompactLinearSlider<false, false, false> tilt_slider_;
        zlgui::attachment::SliderAttachment<true> tilt_attach_;

        juce::Label attack_skew_label_;
        zlgui::slider::CompactLinearSlider<false, false, false> attack_skew_slider_;
        zlgui::attachment::SliderAttachment<true> attack_skew_attach_;

        juce::Label release_skew_label_;
        zlgui::slider::CompactLinearSlider<false, false, false> release_skew_slider_;
        zlgui::attachment::SliderAttachment<true> release_skew_attach_;

        juce::Label gate_label_;
        zlgui::slider::CompactLinearSlider<false, false, false> gate_slider_;
        zlgui::attachment::SliderAttachment<true> gate_attach_;
    };
}

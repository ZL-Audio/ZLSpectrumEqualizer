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

#include "control_background.hpp"

namespace zlpanel {
    class RightControlPanel final : public juce::Component,
                                    private juce::ValueTree::Listener {
    public:
        explicit RightControlPanel(PluginProcessor& p, zlgui::UIBase& base,
                                   const multilingual::TooltipHelper& tooltip_helper);

        ~RightControlPanel() override;

        int getIdealWidth() const;

        void resized() override;

        void repaintCallBackSlow();

        void updateBand();

    private:
        PluginProcessor& p_ref_;
        zlgui::UIBase& base_;
        zlgui::attachment::ComponentUpdater updater_;

        ControlBackground control_background_;

        const std::unique_ptr<juce::Drawable> bypass_drawable_;
        zlgui::button::ClickButton bypass_button_;
        std::unique_ptr<zlgui::attachment::ButtonAttachment<true>> bypass_attachment_;

        zlgui::combobox::CompactCombobox mode_box_;
        std::unique_ptr<zlgui::attachment::ComboBoxAttachment<true>> mode_attachment_;

        zlgui::slider::CompactLinearSlider<true, true, true> th_slider_abs_;
        std::unique_ptr<zlgui::attachment::SliderAttachment<true>> th_attachment_abs_;

        zlgui::slider::CompactLinearSlider<true, true, true> th_slider_band_;
        std::unique_ptr<zlgui::attachment::SliderAttachment<true>> th_attachment_band_;

        zlgui::slider::CompactLinearSlider<true, true, true> th_slider_rel_;
        std::unique_ptr<zlgui::attachment::SliderAttachment<true>> th_attachment_rel_;

        zlgui::slider::CompactLinearSlider<true, true, true> knee_slider_;
        std::unique_ptr<zlgui::attachment::SliderAttachment<true>> knee_attachment_;

        zlgui::slider::CompactLinearSlider<true, true, true> attack_slider_;
        std::unique_ptr<zlgui::attachment::SliderAttachment<true>> attack_attachment_;

        zlgui::slider::CompactLinearSlider<true, true, true> release_slider_;
        std::unique_ptr<zlgui::attachment::SliderAttachment<true>> release_attachment_;

        zlgui::label::NameLookAndFeel label_laf_;

        int c_dynamic_mode_{-1};
    };
}

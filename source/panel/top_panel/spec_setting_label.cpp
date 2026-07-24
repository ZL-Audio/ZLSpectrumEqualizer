// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "spec_setting_label.hpp"

namespace zlpanel {
    SpecSettingLabel::SpecSettingLabel(PluginProcessor& p, zlgui::UIBase& base) :
        base_(base),
        p_ref_(p),
        label_laf_(base),
        resolution_combobox_(zlp::PFFTResolution::kChoices, base, ""),
        resolution_attach_(resolution_combobox_.getBox(), p.parameters_,
                           zlp::PFFTResolution::kID, updater_),
        smooth_slider_("", base, ""),
        smooth_attach_(smooth_slider_.getSlider(), p.parameters_,
                       zlp::PSpecSmooth::kID, updater_) {

        label_laf_.setFontScale(1.5f);

        resolution_label_.setText("Resolution:", juce::dontSendNotification);
        smooth_label_.setText("Smooth:", juce::dontSendNotification);
        for (auto& l : {&resolution_label_, &smooth_label_}) {
            l->setInterceptsMouseClicks(false, false);
            l->setLookAndFeel(&label_laf_);
            l->setJustificationType(juce::Justification::centredRight);
            l->setBufferedToImage(true);
            addAndMakeVisible(l);
        }

        setAlpha(.5f);
        setInterceptsMouseClicks(true, false);
    }

    SpecSettingLabel::~SpecSettingLabel() {
    }

    void SpecSettingLabel::resized() {
        const auto padding = 2 * getPaddingSize(base_.getFontSize());
        const auto bound = getLocalBounds();
    }

    void SpecSettingLabel::repaintCallbackSlow() {

    }

    void SpecSettingLabel::mouseDown(const juce::MouseEvent&) {
        const auto f = static_cast<double>(base_.getPanelProperty(zlgui::PanelSettingIdx::kAnalyzerPanel));
        base_.setPanelProperty(zlgui::PanelSettingIdx::kAnalyzerPanel, f < .5 ? 1. : 0.);
    }

    void SpecSettingLabel::mouseEnter(const juce::MouseEvent&) {
        is_over_ = true;
        const auto f = static_cast<double>(base_.getPanelProperty(zlgui::PanelSettingIdx::kAnalyzerPanel));
        updateAlpha(f > .5);
    }

    void SpecSettingLabel::mouseExit(const juce::MouseEvent&) {
        is_over_ = false;
        const auto f = static_cast<double>(base_.getPanelProperty(zlgui::PanelSettingIdx::kAnalyzerPanel));
        updateAlpha(f > .5);
    }

    void SpecSettingLabel::updateAlpha(const bool is_panel_open) {
        if (is_panel_open) {
            setAlpha(1.f);
        } else if (is_over_) {
            setAlpha(.75f);
        } else {
            setAlpha(.5f);
        }
    }
}

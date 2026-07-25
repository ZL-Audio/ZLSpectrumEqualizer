// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "spec_setting_panel.hpp"

namespace zlpanel {
    SpecSettingPanel::SpecSettingPanel(PluginProcessor& p, zlgui::UIBase& base,
                                       const multilingual::TooltipHelper&) :
        base_(base),
        spec_setting_open_(*p.parameters_NA_.getRawParameterValue(zlstate::PSpecSettingOpen::kID)),
        control_background_(base),
        label_laf_(base),
        tilt_label_("", "Tilt"),
        tilt_slider_("", base, ""),
        tilt_attach_(tilt_slider_.getSlider(), p.parameters_,
                     zlp::PSpecTilt::kID, updater_),
        attack_skew_label_("", "Attack Skew"),
        attack_skew_slider_("", base, ""),
        attack_skew_attach_(attack_skew_slider_.getSlider(), p.parameters_,
                            zlp::PSpecSkewAttack::kID, updater_),
        release_skew_label_("", "Release Skew"),
        release_skew_slider_("", base, ""),
        release_skew_attach_(release_skew_slider_.getSlider(), p.parameters_,
                             zlp::PSpecSkewRelease::kID, updater_),
        gate_label_("", "Gate"),
        gate_slider_("", base, ""),
        gate_attach_(gate_slider_.getSlider(), p.parameters_,
                     zlp::PSpecGate::kID, updater_) {
        control_background_.setBufferedToImage(true);
        addAndMakeVisible(control_background_);

        label_laf_.setFontScale(1.5f);
        for (auto& l : {&tilt_label_, &attack_skew_label_, &release_skew_label_, &gate_label_}) {
            l->setJustificationType(juce::Justification::centredRight);
            l->setLookAndFeel(&label_laf_);
            l->setBufferedToImage(true);
            addAndMakeVisible(l);
        }

        for (auto& s : {&tilt_slider_, &attack_skew_slider_, &release_skew_slider_, &gate_slider_}) {
            s->getSlider().setSliderSnapsToMousePosition(false);
            s->setBufferedToImage(true);
            addAndMakeVisible(s);
        }

        setBufferedToImage(true);
    }

    int SpecSettingPanel::getIdealWidth() const {
        const auto font_size = base_.getFontSize();
        const auto padding = getPaddingSize(font_size);
        const auto slider_width = getSliderWidth(font_size);
        const auto button_height = getButtonSize(font_size);

        return 5 * padding + 8 * slider_width - 4 * button_height;
    }

    int SpecSettingPanel::getIdealHeight() const {
        const auto font_size = base_.getFontSize();
        const auto padding = getPaddingSize(font_size);
        const auto button_height = getButtonSize(font_size);

        return 2 * padding + button_height;
    }

    void SpecSettingPanel::resized() {
        const auto font_size = base_.getFontSize();
        const auto button_height = getButtonSize(font_size);
        const auto slider_width = getSliderWidth(font_size);
        const auto padding = getPaddingSize(font_size);

        auto bound = getLocalBounds();
        control_background_.setBounds(bound);

        {
            bound.removeFromLeft(padding);
            tilt_label_.setBounds(bound.removeFromLeft(slider_width - button_height));
            tilt_slider_.setBounds(bound.removeFromLeft(slider_width - button_height));
        }
        {
            bound.removeFromLeft(padding);
            attack_skew_label_.setBounds(bound.removeFromLeft(slider_width + button_height));
            attack_skew_slider_.setBounds(bound.removeFromLeft(slider_width - button_height));
        }
        {
            bound.removeFromLeft(padding);
            release_skew_label_.setBounds(bound.removeFromLeft(slider_width + button_height));
            release_skew_slider_.setBounds(bound.removeFromLeft(slider_width - button_height));
        }
        {
            bound.removeFromLeft(padding);
            gate_label_.setBounds(bound.removeFromLeft(slider_width - button_height));
            gate_slider_.setBounds(bound.removeFromLeft(slider_width - button_height));
        }

        const auto dragging_distance = getSliderDraggingDistance(font_size);
        for (auto& s : {&tilt_slider_, &attack_skew_slider_, &release_skew_slider_, &gate_slider_}) {
            s->setMouseDragSensitivity(dragging_distance);
        }
    }

    void SpecSettingPanel::repaintCallBackSlow() {
        updater_.updateComponents();
        setVisible(spec_setting_open_.load(std::memory_order::relaxed) > .5f);
    }
}

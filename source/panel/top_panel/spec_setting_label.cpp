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
        resolution_box_(zlp::PSpecResolution::kChoices, base, ""),
        resolution_attach_(resolution_box_.getBox(), p.parameters_,
                           zlp::PSpecResolution::kID, updater_),
        smooth_type_box_(zlp::PSpecSmoothType::kChoices, base, ""),
        smooth_type_attach_(smooth_type_box_.getBox(), p.parameters_,
                            zlp::PSpecSmoothType::kID, updater_),
        smooth_slider_("", base, ""),
        smooth_attach_(smooth_slider_.getSlider(), p.parameters_,
                       zlp::PSpecSmooth::kID, updater_) {
        const auto popup_option = juce::PopupMenu::Options().withPreferredPopupDirection(
            juce::PopupMenu::Options::PopupDirection::downwards);

        for (auto& box: {&resolution_box_, &smooth_type_box_}) {
            box->getLAF().setOption(popup_option);
            box->setBufferedToImage(true);
            addAndMakeVisible(box);
        }

        smooth_slider_.getSlider().setSliderSnapsToMousePosition(false);
        smooth_slider_.setBufferedToImage(true);
        addAndMakeVisible(smooth_slider_);

        setAlpha(.5f);
        setInterceptsMouseClicks(true, false);
    }

    SpecSettingLabel::~SpecSettingLabel() {
    }

    void SpecSettingLabel::resized() {
        const auto font_size = base_.getFontSize();
        const auto padding = getPaddingSize(font_size);
        const auto button_height = getButtonSize(font_size);
        const auto slider_width = getSliderWidth(font_size);

        auto right_bound = getLocalBounds();
        auto left_bound = right_bound.removeFromLeft(right_bound.getWidth() / 2);

        left_bound.removeFromRight(button_height / 2 + padding);
        resolution_box_.setBounds(left_bound.removeFromRight(slider_width + button_height));

        right_bound.removeFromLeft(button_height / 2 + padding);
        smooth_type_box_.setBounds(right_bound.removeFromLeft(button_height * 2));
        smooth_slider_.setBounds(right_bound.removeFromLeft(slider_width - button_height));

        const auto dragging_distance = getSliderDraggingDistance(font_size);
        smooth_slider_.setMouseDragSensitivity(dragging_distance);
    }

    void SpecSettingLabel::repaintCallbackSlow() {
        updater_.updateComponents();
    }

    void SpecSettingLabel::mouseDown(const juce::MouseEvent&) {
        // const auto f = static_cast<double>(base_.getPanelProperty(zlgui::PanelSettingIdx::kAnalyzerPanel));
        // base_.setPanelProperty(zlgui::PanelSettingIdx::kAnalyzerPanel, f < .5 ? 1. : 0.);
    }

    void SpecSettingLabel::mouseEnter(const juce::MouseEvent&) {
        is_over_ = true;
        // const auto f = static_cast<double>(base_.getPanelProperty(zlgui::PanelSettingIdx::kAnalyzerPanel));
        // updateAlpha(f > .5);
    }

    void SpecSettingLabel::mouseExit(const juce::MouseEvent&) {
        is_over_ = false;
        // const auto f = static_cast<double>(base_.getPanelProperty(zlgui::PanelSettingIdx::kAnalyzerPanel));
        // updateAlpha(f > .5);
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

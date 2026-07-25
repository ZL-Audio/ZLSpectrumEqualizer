// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "spec_setting_label.hpp"
#include <BinaryData.h>

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
                       zlp::PSpecSmooth::kID, updater_),
        setting_drawable_(juce::Drawable::createFromImageData(BinaryData::settings_svg,
                                                                   BinaryData::settings_svgSize)),
        setting_button_(base, setting_drawable_.get(), setting_drawable_.get(), ""),
        setting_attach_(setting_button_.getButton(), p.parameters_NA_,
                             zlstate::PSpecSettingOpen::kID, updater_) {
        const auto popup_option = juce::PopupMenu::Options().withPreferredPopupDirection(
            juce::PopupMenu::Options::PopupDirection::downwards);

        resolution_box_.getLAF().setItemJustification(juce::Justification::centredRight);
        resolution_box_.getLAF().setLabelJustification(juce::Justification::centredRight);
        smooth_type_box_.getLAF().setItemJustification(juce::Justification::centredLeft);
        smooth_type_box_.getLAF().setLabelJustification(juce::Justification::centredLeft);
        for (auto& box : {&resolution_box_, &smooth_type_box_}) {
            box->getLAF().setOption(popup_option);
            box->setBufferedToImage(true);
            addAndMakeVisible(box);
        }

        smooth_slider_.getSlider().setSliderSnapsToMousePosition(false);
        smooth_slider_.setBufferedToImage(true);
        addAndMakeVisible(smooth_slider_);

        setting_button_.setBufferedToImage(true);
        addAndMakeVisible(setting_button_);

        setComponentsAlpha(.5f);
        setInterceptsMouseClicks(false, true);
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
        resolution_box_.setBounds(left_bound.removeFromRight(slider_width + button_height + padding));

        right_bound.removeFromLeft(button_height / 2 + padding);
        smooth_type_box_.setBounds(right_bound.removeFromLeft(button_height * 2));
        smooth_slider_.setBounds(right_bound.removeFromLeft(slider_width - button_height));

        setting_button_.setBounds(getLocalBounds().withSizeKeepingCentre(button_height, button_height));

        const auto dragging_distance = getSliderDraggingDistance(font_size);
        smooth_slider_.setMouseDragSensitivity(dragging_distance);
    }

    void SpecSettingLabel::repaintCallbackSlow() {
        updater_.updateComponents();
        if (setting_button_.getButton().getToggleState()) {
            setComponentsAlpha(1.f);
        } else {
            setComponentsAlpha(.5f);
        }
    }

    void SpecSettingLabel::setComponentsAlpha(const float alpha) {
        const auto c_alpha = resolution_box_.getAlpha();
        if (std::abs(c_alpha - alpha) > 0.01f) {
            resolution_box_.setAlpha(alpha);
            smooth_type_box_.setAlpha(alpha);
            smooth_slider_.setAlpha(alpha);
        }
    }
}

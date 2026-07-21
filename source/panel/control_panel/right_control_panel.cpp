// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "right_control_panel.hpp"
#include "BinaryData.h"

namespace zlpanel {
    RightControlPanel::RightControlPanel(PluginProcessor& p,
                                         zlgui::UIBase& base,
                                         const multilingual::TooltipHelper& tooltip_helper) :
        p_ref_(p), base_(base), updater_(),
        control_background_(base),
        bypass_drawable_(juce::Drawable::createFromImageData(BinaryData::bypass_svg,
                                                             BinaryData::bypass_svgSize)),
        bypass_button_(base, bypass_drawable_.get(), bypass_drawable_.get(),
                       tooltip_helper.getToolTipText(multilingual::kBandDynamicBypass)),
        mode_box_(zlp::PDynamicMode::kChoices, base,
                  "Dynamic Mode"),
        th_slider_abs_("Threshold", base,
                       tooltip_helper.getToolTipText(multilingual::kBandDynamicThreshold)),
        th_slider_band_("Threshold", base,
                        tooltip_helper.getToolTipText(multilingual::kBandDynamicThreshold)),
        th_slider_rel_("Threshold", base,
                       tooltip_helper.getToolTipText(multilingual::kBandDynamicThreshold)),
        knee_slider_("Knee", base,
                     tooltip_helper.getToolTipText(multilingual::kBandDynamicThreshold)),
        attack_slider_("Attack", base,
                       tooltip_helper.getToolTipText(multilingual::kBandDynamicAttack)),
        release_slider_("Release", base,
                        tooltip_helper.getToolTipText(multilingual::kBandDynamicRelease)),
        label_laf_(base) {
        control_background_.setBufferedToImage(true);
        addAndMakeVisible(control_background_);

        bypass_button_.setImageAlpha(1.f, 1.f, .5f, .75f);
        bypass_button_.setBufferedToImage(true);
        addAndMakeVisible(bypass_button_);

        mode_box_.setBufferedToImage(true);
        addAndMakeVisible(mode_box_);

        th_slider_abs_.setBufferedToImage(true);
        addChildComponent(th_slider_abs_);
        th_slider_band_.setBufferedToImage(true);
        addChildComponent(th_slider_band_);
        th_slider_rel_.setBufferedToImage(true);
        addChildComponent(th_slider_rel_);

        knee_slider_.setBufferedToImage(true);
        addAndMakeVisible(knee_slider_);

        attack_slider_.setBufferedToImage(true);
        addAndMakeVisible(attack_slider_);

        release_slider_.setBufferedToImage(true);
        addAndMakeVisible(release_slider_);
    }

    RightControlPanel::~RightControlPanel() {
    }

    int RightControlPanel::getIdealWidth() const {
        const auto font_size = base_.getFontSize();
        const auto slider_width = getSliderWidth(font_size);
        const auto padding = getPaddingSize(font_size);

        return 3 * padding + 2 * (padding / 2) + 2 * slider_width;
    }

    void RightControlPanel::resized() {
        const auto font_size = base_.getFontSize();
        const auto button_height = getButtonSize(font_size);
        const auto slider_width = getSliderWidth(font_size);
        const auto slider_height = getSliderHeight(font_size);
        const auto padding = getPaddingSize(font_size);

        control_background_.setBounds(getLocalBounds());

        auto bound = getLocalBounds();
        bound.reduce(padding + padding / 2, padding);

        {
            auto temp_bound = bound.removeFromTop(button_height);
            bypass_button_.setBounds(temp_bound.removeFromLeft(button_height));
            temp_bound.removeFromLeft(padding);
            mode_box_.setBounds(temp_bound);
        }

        const auto h_padding = (bound.getHeight() - 2 * slider_height) / 4;
        {
            auto temp_bound = bound.removeFromLeft(slider_width);
            temp_bound.removeFromBottom(h_padding);
            knee_slider_.setBounds(temp_bound.removeFromBottom(slider_height));
            temp_bound.removeFromBottom(2 * h_padding);
            th_slider_abs_.setBounds(temp_bound.withBottom(temp_bound.getBottom()));
            th_slider_band_.setBounds(temp_bound.withBottom(temp_bound.getBottom()));
            th_slider_rel_.setBounds(temp_bound.withBottom(temp_bound.getBottom()));

            th_slider_abs_.setBounds(temp_bound.removeFromBottom(slider_height));
            th_slider_band_.setBounds(th_slider_abs_.getBounds());
            th_slider_rel_.setBounds(th_slider_abs_.getBounds());
        }
        bound.removeFromLeft(padding);
        {
            auto temp_bound = bound.removeFromLeft(slider_width);
            temp_bound.removeFromBottom(h_padding);
            release_slider_.setBounds(temp_bound.removeFromBottom(slider_height));
            temp_bound.removeFromBottom(2 * h_padding);
            attack_slider_.setBounds(temp_bound.removeFromBottom(slider_height));
        }

        const auto dragging_distance = getSliderDraggingDistance(font_size);
        th_slider_abs_.setMouseDragSensitivity(dragging_distance);
        th_slider_band_.setMouseDragSensitivity(dragging_distance);
        th_slider_rel_.setMouseDragSensitivity(dragging_distance);
        knee_slider_.setMouseDragSensitivity(dragging_distance);
        attack_slider_.setMouseDragSensitivity(dragging_distance);
        release_slider_.setMouseDragSensitivity(dragging_distance);
    }

    void RightControlPanel::repaintCallBackSlow() {
        updater_.updateComponents();
        if (c_dynamic_mode_ != mode_box_.getBox().getSelectedItemIndex()) {
            c_dynamic_mode_ = mode_box_.getBox().getSelectedItemIndex();
            th_slider_abs_.setVisible(c_dynamic_mode_ == 0);
            th_slider_band_.setVisible(c_dynamic_mode_ == 1);
            th_slider_rel_.setVisible(c_dynamic_mode_ == 2);
        }
    }

    void RightControlPanel::updateBand() {
        if (base_.getSelectedBand() < zlp::kBandNum) {
            const auto band_s = std::to_string(base_.getSelectedBand());
            bypass_attachment_ = std::make_unique<zlgui::attachment::ButtonAttachment<true>>(
                bypass_button_.getButton(), p_ref_.parameters_, zlp::PDynamicBypass::kID + band_s, updater_);
            mode_attachment_ = std::make_unique<zlgui::attachment::ComboBoxAttachment<true>>(
                mode_box_.getBox(), p_ref_.parameters_, zlp::PDynamicMode::kID + band_s, updater_);
            th_attachment_abs_ = std::make_unique<zlgui::attachment::SliderAttachment<true>>(
                th_slider_abs_.getSlider(), p_ref_.parameters_, zlp::PThresholdAbs::kID + band_s, updater_);
            th_slider_abs_.setComponentID(zlp::PThresholdAbs::kID + band_s);
            th_attachment_band_ = std::make_unique<zlgui::attachment::SliderAttachment<true>>(
                th_slider_band_.getSlider(), p_ref_.parameters_, zlp::PThresholdBand::kID + band_s, updater_);
            th_slider_band_.setComponentID(zlp::PThresholdBand::kID + band_s);
            th_attachment_rel_ = std::make_unique<zlgui::attachment::SliderAttachment<true>>(
                th_slider_rel_.getSlider(), p_ref_.parameters_, zlp::PThresholdRel::kID + band_s, updater_);
            th_slider_rel_.setComponentID(zlp::PThresholdRel::kID + band_s);
            knee_attachment_ = std::make_unique<zlgui::attachment::SliderAttachment<true>>(
                knee_slider_.getSlider(), p_ref_.parameters_, zlp::PKneeW::kID + band_s, updater_);
            knee_slider_.setComponentID(zlp::PKneeW::kID + band_s);
            attack_attachment_ = std::make_unique<zlgui::attachment::SliderAttachment<true>>(
                attack_slider_.getSlider(), p_ref_.parameters_, zlp::PAttack::kID + band_s, updater_);
            attack_slider_.setComponentID(zlp::PAttack::kID + band_s);
            release_attachment_ = std::make_unique<zlgui::attachment::SliderAttachment<true>>(
                release_slider_.getSlider(), p_ref_.parameters_, zlp::PRelease::kID + band_s, updater_);
            release_slider_.setComponentID(zlp::PRelease::kID + band_s);
        } else {
            bypass_attachment_.reset();
            mode_attachment_.reset();
            th_attachment_abs_.reset();
            th_attachment_band_.reset();
            th_attachment_rel_.reset();
            knee_attachment_.reset();
            attack_attachment_.reset();
            release_attachment_.reset();
        }
    }
}

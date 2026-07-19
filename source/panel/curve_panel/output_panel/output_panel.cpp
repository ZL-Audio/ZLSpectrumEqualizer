// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "output_panel.hpp"
#include "BinaryData.h"

namespace zlpanel {
    OutputPanel::OutputPanel(PluginProcessor& p, zlgui::UIBase& base,
                             const multilingual::TooltipHelper& tooltip_helper) :
        p_ref_(p), base_(base), updater_(),
        control_background_(base),
        name_laf_(base),
        gain_label_("", "GAIN"),
        scale_label_("", "SCALE"),
        gain_slider_("", base_,
                     tooltip_helper.getToolTipText(multilingual::kOutputGain), 1.25f),
        gain_attach_(gain_slider_.getSlider1(), p.parameters_,
                     zlp::POutputGain::kID, updater_),
        scale_slider_("", base,
                      tooltip_helper.getToolTipText(multilingual::kGainScale), 1.25f),
        scale_attach_(scale_slider_.getSlider1(), p.parameters_,
                      zlp::PGainScale::kID, updater_),
        sgc_drawable_(juce::Drawable::createFromImageData(BinaryData::dline_s_svg,
                                                          BinaryData::dline_s_svgSize)),
        sgc_button_(base, sgc_drawable_.get(), sgc_drawable_.get(),
                    tooltip_helper.getToolTipText(multilingual::kStaticGC)),
        sgc_attach_(sgc_button_.getButton(), p.parameters_,
                    zlp::PStaticGain::kID, updater_) {
        juce::ignoreUnused(p_ref_, base_, tooltip_helper);

        control_background_.setBufferedToImage(true);
        addAndMakeVisible(control_background_);

        name_laf_.setFontScale(1.5f);

        gain_label_.setLookAndFeel(&name_laf_);
        gain_label_.setJustificationType(juce::Justification::centred);
        gain_label_.setBufferedToImage(true);
        addAndMakeVisible(gain_label_);

        scale_label_.setLookAndFeel(&name_laf_);
        scale_label_.setJustificationType(juce::Justification::centred);
        scale_label_.setBufferedToImage(true);
        addAndMakeVisible(scale_label_);

        gain_slider_.setComponentID(zlp::POutputGain::kID);
        gain_slider_.setBufferedToImage(true);
        addAndMakeVisible(gain_slider_);

        scale_slider_.setComponentID(zlp::PGainScale::kID);
        scale_slider_.setBufferedToImage(true);
        addAndMakeVisible(scale_slider_);

        sgc_button_.setImageAlpha(.5f, .75f, 1.f, 1.f);
        sgc_button_.setBufferedToImage(true);
        addAndMakeVisible(&sgc_button_);

        base_.setPanelProperty(zlgui::PanelSettingIdx::kOutputPanel, 0.);
        base_.getPanelValueTree().addListener(this);
    }

    OutputPanel::~OutputPanel() {
        base_.getPanelValueTree().removeListener(this);
    }

    int OutputPanel::getIdealWidth() const {
        const auto font_size = base_.getFontSize();
        const auto slider_width = getSliderWidth(font_size);
        const auto padding = getPaddingSize(font_size);
        return 5 * padding + 2 * slider_width;
    }

    int OutputPanel::getIdealHeight() const {
        const auto font_size = base_.getFontSize();
        const auto slider_width = getSliderWidth(font_size);
        const auto button_height = getButtonSize(font_size);
        const auto padding = getPaddingSize(font_size);
        return 5 * padding + 3 * button_height + slider_width;
    }

    void OutputPanel::resized() {
        const auto font_size = base_.getFontSize();
        const auto slider_width = getSliderWidth(font_size);
        const auto button_height = getButtonSize(font_size);
        const auto padding = getPaddingSize(font_size);

        auto bound = getLocalBounds();
        control_background_.setBounds(bound);

        bound.reduce(2 * padding, padding);
        {
            auto t_bound = bound.removeFromTop(button_height);
            gain_label_.setBounds(t_bound.removeFromLeft(slider_width));
            scale_label_.setBounds(t_bound.removeFromRight(slider_width));
        }
        bound.removeFromTop(padding);
        {
            auto t_bound = bound.removeFromTop(slider_width);
            gain_slider_.setBounds(t_bound.removeFromLeft(slider_width));
            scale_slider_.setBounds(t_bound.removeFromRight(slider_width));
        }
        bound.removeFromTop(padding);
        {
            auto t_bound = bound.removeFromTop(button_height);
            sgc_button_.setBounds(t_bound.withSizeKeepingCentre(button_height, button_height));
        }

        const auto dragging_distance = getSliderDraggingDistance(font_size);
        gain_slider_.setMouseDragSensitivity(dragging_distance);
        scale_slider_.setMouseDragSensitivity(dragging_distance);
    }

    void OutputPanel::repaintCallBackSlow() {
        updater_.updateComponents();
    }

    void OutputPanel::valueTreePropertyChanged(juce::ValueTree&, const juce::Identifier& property) {
        if (base_.isPanelIdentifier(zlgui::PanelSettingIdx::kOutputPanel, property)) {
            setVisible(static_cast<double>(base_.getPanelProperty(zlgui::PanelSettingIdx::kOutputPanel)) > .5);
        }
    }
}

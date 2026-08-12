// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "top_panel.hpp"
#include "BinaryData.h"

namespace zlpanel {
    TopPanel::TopPanel(PluginProcessor& p, zlgui::UIBase& base,
                       const multilingual::TooltipHelper& tooltip_helper) :
        p_ref_(p), base_(base), updater_(),
        logo_panel_(p, base, tooltip_helper),
        output_label_(p, base),
        analyzer_label_(p, base),
        spec_setting_label_(p, base, tooltip_helper),
        preset_drawable_(juce::Drawable::createFromImageData(BinaryData::collections_bookmark_svg,
                                                             BinaryData::collections_bookmark_svgSize)),
        preset_button_(base, preset_drawable_.get(), nullptr, ""),
        bypass_drawable_(juce::Drawable::createFromImageData(BinaryData::bypass_svg,
                                                             BinaryData::bypass_svgSize)),
        bypass_button_(base, bypass_drawable_.get(), bypass_drawable_.get(),
                       tooltip_helper.getToolTipText(multilingual::kBypass)),
        bypass_attach_(bypass_button_.getButton(), p.parameters_, zlp::PBypass::kID, updater_),
        ext_drawable_(juce::Drawable::createFromImageData(BinaryData::external_side_svg,
                                                          BinaryData::external_side_svgSize)),
        ext_button_(base, ext_drawable_.get(), ext_drawable_.get(),
                    tooltip_helper.getToolTipText(multilingual::kExternalSideChain)),
        ext_attach_(ext_button_.getButton(), p.parameters_, zlp::PExtSide::kID, updater_) {
        logo_panel_.setBufferedToImage(true);
        addAndMakeVisible(logo_panel_);

        output_label_.setBufferedToImage(true);
        addAndMakeVisible(output_label_);

        analyzer_label_.setBufferedToImage(true);
        addAndMakeVisible(analyzer_label_);

        addAndMakeVisible(spec_setting_label_);

        bypass_button_.setImageAlpha(1.f, 1.f, .5f, .75f);
        bypass_button_.setBufferedToImage(true);
        addAndMakeVisible(bypass_button_);

        preset_button_.setImageAlpha(.5f, .75f);
        preset_button_.setBufferedToImage(true);
        preset_button_.getButton().setTitle("Open preset browser");
        preset_button_.getButton().onClick = [this]() {
            const auto preset_open = static_cast<float>(base_.getPanelProperty(zlgui::PanelSettingIdx::kPresetBrowser));
            base_.setPanelProperty(zlgui::PanelSettingIdx::kPresetBrowser, preset_open < .5f ? 1.f : 0.f);
        };
        addAndMakeVisible(preset_button_);

        ext_button_.getButton().onClick = [this]() {
            if (ext_button_.getToggleState()) {
                auto* para = p_ref_.parameters_NA_.getParameter(zlstate::PFFTSideON::kID);
                updateValue(para, 1.f);
            }
        };
        ext_button_.setImageAlpha(.5f, .75f, 1.f, 1.f);
        ext_button_.setBufferedToImage(true);
        addAndMakeVisible(ext_button_);

        setInterceptsMouseClicks(false, true);
    }

    void TopPanel::paint(juce::Graphics& g) {
        g.fillAll(base_.getBackgroundColour());
    }

    int TopPanel::getIdealHeight() const {
        const auto font_size = base_.getFontSize();
        return 2 * (getPaddingSize(font_size) / 2) + getButtonSize(font_size);
    }

    void TopPanel::resized() {
        const auto font_size = base_.getFontSize();
        const auto padding = getPaddingSize(font_size);
        const auto slider_width = getSliderWidth(font_size);

        auto bound = getLocalBounds();
        bound.reduce(padding / 2, padding / 2);

        logo_panel_.setBounds(bound.removeFromLeft(bound.getHeight() * 2 + padding));
        bound.removeFromLeft(padding);

        bypass_button_.setBounds(bound.removeFromRight(bound.getHeight()));
        bound.removeFromRight(padding);
        ext_button_.setBounds(bound.removeFromRight(bound.getHeight()));
        {
            const auto left_pad = bound.getX();
            const auto t_width = 6 * padding + 3 * (slider_width / 2) - left_pad;
            bound.removeFromLeft(padding);
            preset_button_.setBounds(bound.removeFromLeft(bound.getHeight()));
            preset_button_.getButton().setEdgeIndent(static_cast<int>(std::round(font_size * .15f)));
            bound.removeFromLeft(padding);
            analyzer_label_.setBounds(bound.getX(), 0, t_width, getHeight());
            bound.removeFromLeft(t_width);
        }
        {
            const auto right_pad = getWidth() - bound.getRight();
            const auto t_width = 5 * padding + 2 * slider_width - right_pad + 2 * padding;
            output_label_.setBounds(bound.getRight() - t_width, 0, t_width, getHeight());
            bound.removeFromRight(t_width);
        }
        {
            auto t_bound = getLocalBounds();
            const auto t1 = bound.getX() - t_bound.getX();
            const auto t2 = t_bound.getRight() - bound.getRight();
            t_bound.reduce(std::max(t1, t2), padding / 2);
            spec_setting_label_.setBounds(t_bound);
        }
    }

    void TopPanel::repaintCallbackSlow() {
        output_label_.repaintCallbackSlow();
        spec_setting_label_.repaintCallbackSlow();
        updater_.updateComponents();
    }
}

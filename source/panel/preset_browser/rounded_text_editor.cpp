// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "rounded_text_editor.hpp"
#include "preset_style.hpp"

namespace zlpanel {
    RoundedTextEditor::RoundedTextEditor(zlgui::UIBase& base) : base_(base) {
        setColour(backgroundColourId, juce::Colours::transparentBlack);
        setColour(outlineColourId, juce::Colours::transparentBlack);
        setColour(focusedOutlineColourId, juce::Colours::transparentBlack);
    }

    void RoundedTextEditor::paint(juce::Graphics& g) {
        const auto font_size = base_.getFontSize();
        const auto corner = font_size * .45f;
        const auto outline = font_size * .08f;
        const auto bounds = getLocalBounds().toFloat();
        g.setColour(preset_style::surfaceColour(base_.getBackgroundColour(), base_.getTextColour()));
        g.fillRoundedRectangle(bounds, corner);
        if (hasKeyboardFocus(true)) {
            g.setColour(base_.getTextColour().withAlpha(.24f));
            g.drawRoundedRectangle(bounds.reduced(outline * .5f), corner, outline);
        }
        juce::TextEditor::paint(g);
    }

    void RoundedTextEditor::lookAndFeelChanged() {
        setColour(backgroundColourId, juce::Colours::transparentBlack);
        setColour(textColourId, base_.getTextColour());
        setColour(highlightColourId, base_.getTextColour().withAlpha(.22f));
        setColour(highlightedTextColourId, base_.getTextColour());
        setColour(outlineColourId, juce::Colours::transparentBlack);
        setColour(focusedOutlineColourId, juce::Colours::transparentBlack);
        juce::TextEditor::lookAndFeelChanged();
    }
}

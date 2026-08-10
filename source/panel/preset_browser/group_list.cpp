// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "group_list.hpp"
#include "preset_style.hpp"

namespace zlpanel {
    GroupList::GroupList(zlgui::UIBase& base) : VirtualizedList(base) {
    }

    void GroupList::setGroups(const juce::StringArray& groups, const juce::String& selected_group) {
        groups_ = groups;
        setRowCount(groups_.size());
        setSelectedRow(groups_.indexOf(selected_group));
    }

    void GroupList::paintRow(juce::Graphics& g, const int row, juce::Rectangle<int> bounds,
                             const bool selected, const bool hovered) {
        if (!juce::isPositiveAndBelow(row, groups_.size())) {
            return;
        }

        const auto font_size = getBase().getFontSize();
        auto card = bounds.toFloat().reduced(font_size * .16f);
        if (selected || hovered) {
            g.setColour(getBase().getTextColour().withAlpha(selected ? .115f : .05f));
            g.fillRoundedRectangle(card, font_size * .35f);
        }
        if (selected) {
            const auto accent_width = font_size * .24f;
            auto accent = card.removeFromLeft(accent_width)
                .withSizeKeepingCentre(accent_width, card.getHeight() * .48f);
            g.setColour(getBase().getTextColour().withAlpha(.72f));
            g.fillRoundedRectangle(accent, font_size * .12f);
        }

        g.setColour(getBase().getTextColour().withAlpha(selected ? .95f : .68f));
        g.setFont(juce::FontOptions{preset_style::textFontSize(font_size)});
        g.drawFittedText(groups_[row], bounds.reduced(preset_list_layout::rowTextInset(font_size), 0),
                         juce::Justification::centredLeft, 1);
    }

    void GroupList::rowClicked(const int row) {
        if (juce::isPositiveAndBelow(row, groups_.size()) && onGroupSelected) {
            onGroupSelected(groups_[row]);
        }
    }

    void GroupList::rowDoubleClicked(const int row) {
        rowClicked(row);
    }
}

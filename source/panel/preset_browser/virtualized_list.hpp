// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

#include "../../gui/interface_definitions.hpp"

namespace zlpanel {
    namespace preset_list_layout {
        inline int contentInset(const float font_size) {
            return juce::roundToInt(font_size * .24f);
        }

        inline int rowTextInset(const float font_size) {
            return juce::roundToInt(font_size * .96f);
        }

        inline int columnTextInset(const float font_size) {
            return contentInset(font_size) + rowTextInset(font_size);
        }
    }

    class VirtualizedList : public juce::Component,
                            private juce::ScrollBar::Listener {
    public:
        explicit VirtualizedList(zlgui::UIBase& base);

        ~VirtualizedList() override;

        void paint(juce::Graphics& g) override;

        void resized() override;

        void mouseMove(const juce::MouseEvent& event) override;

        void mouseExit(const juce::MouseEvent&) override;

        void mouseDown(const juce::MouseEvent& event) override;

        void mouseUp(const juce::MouseEvent& event) override;

        void mouseDoubleClick(const juce::MouseEvent& event) override;

        void mouseWheelMove(const juce::MouseEvent&, const juce::MouseWheelDetails& wheel) override;

        void lookAndFeelChanged() override;

        void setRowCount(int row_count);

        void setRowHeight(int row_height);

        void setSelectedRow(int row);

        void scrollToRow(int row);

    protected:
        virtual void paintRow(juce::Graphics& g, int row, juce::Rectangle<int> bounds,
                              bool selected, bool hovered) = 0;

        virtual void rowClicked(int row) = 0;

        virtual void rowDoubleClicked(int row) = 0;

        zlgui::UIBase& getBase() const;

    private:
        zlgui::UIBase& base_;
        juce::ScrollBar scroll_bar_{true};
        int row_count_{0};
        int row_height_{1};
        int selected_row_{-1};
        int hovered_row_{-1};
        int mouse_down_row_{-1};
        double scroll_position_{0.0};

        juce::Rectangle<int> getContentBounds() const;

        int getRowAt(juce::Point<int> position) const;

        void setScrollPosition(double position);

        void updateScrollRange();

        void scrollBarMoved(juce::ScrollBar*, double new_range_start) override;
    };
}

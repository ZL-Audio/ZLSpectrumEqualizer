// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <array>
#include <functional>
#include <vector>

#include <juce_gui_basics/juce_gui_basics.h>

#include "../../gui/interface_definitions.hpp"

namespace zlpanel {
    class UISettingText final : public juce::Component {
    public:
        UISettingText(zlgui::UIBase& base, juce::String text,
                      float font_scale, float alpha,
                      juce::Justification justification = juce::Justification::centredLeft);

        void paint(juce::Graphics& g) override;

    private:
        zlgui::UIBase& base_;
        juce::String text_;
        float font_scale_;
        float alpha_;
        juce::Justification justification_;
    };

    class UISettingPanelBackground final : public juce::Component {
    public:
        explicit UISettingPanelBackground(zlgui::UIBase& base);

        void paint(juce::Graphics& g) override;

        void setSurfaceBounds(std::vector<juce::Rectangle<int>> bounds);

    private:
        zlgui::UIBase& base_;
        std::vector<juce::Rectangle<int>> surface_bounds_;
    };

    class UISettingTabBar final : public juce::Component {
    public:
        explicit UISettingTabBar(zlgui::UIBase& base);

        void paint(juce::Graphics& g) override;

        void mouseMove(const juce::MouseEvent& event) override;

        void mouseExit(const juce::MouseEvent& event) override;

        void mouseDown(const juce::MouseEvent& event) override;

        void mouseUp(const juce::MouseEvent& event) override;

        bool keyPressed(const juce::KeyPress& key) override;

        void setSelectedIndex(int index);

        std::function<void(int)> onTabSelected;

    private:
        zlgui::UIBase& base_;
        const std::array<juce::String, 4> tab_names_{"Colour", "Control", "Other", "Credit"};
        int selected_index_{0};
        int hovered_index_{-1};
        int mouse_down_index_{-1};

        [[nodiscard]] juce::Rectangle<int> getTabBounds(int index) const;

        [[nodiscard]] int getTabAt(juce::Point<int> position) const;

        void selectTab(int index, bool send_notification);
    };

    class UISettingViewport final : public juce::Component {
    public:
        explicit UISettingViewport(zlgui::UIBase& base);

        ~UISettingViewport() override;

        void paint(juce::Graphics& g) override;

        void resized() override;

        void mouseMove(const juce::MouseEvent& event) override;

        void mouseExit(const juce::MouseEvent& event) override;

        void mouseDown(const juce::MouseEvent& event) override;

        void mouseDrag(const juce::MouseEvent& event) override;

        void mouseUp(const juce::MouseEvent& event) override;

        void mouseWheelMove(const juce::MouseEvent& event,
                            const juce::MouseWheelDetails& wheel) override;

        bool keyPressed(const juce::KeyPress& key) override;

        void setViewedComponent(juce::Component* component, int content_height);

        void setViewPosition(double position);

        [[nodiscard]] double getViewPosition() const;

    private:
        zlgui::UIBase& base_;
        juce::Component* viewed_component_{nullptr};
        int content_height_{0};
        double view_position_{0.0};
        bool scroll_bar_hovered_{false};
        bool scroll_bar_dragging_{false};
        int drag_offset_{0};

        [[nodiscard]] juce::Rectangle<int> getContentBounds() const;

        [[nodiscard]] juce::Rectangle<int> getScrollTrackBounds() const;

        [[nodiscard]] juce::Rectangle<int> getScrollThumbBounds() const;

        [[nodiscard]] double getMaximumViewPosition() const;

        [[nodiscard]] bool needsScrollBar() const;

        void updateViewedComponentBounds();
    };
}

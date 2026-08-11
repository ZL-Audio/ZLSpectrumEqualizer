// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "ui_setting_components.hpp"

#include <cmath>
#include <utility>

#include "../helper/helper.hpp"
#include "../preset_browser/preset_style.hpp"

namespace zlpanel {
    UISettingText::UISettingText(zlgui::UIBase& base, juce::String text,
                                 const float font_scale, const float alpha,
                                 const juce::Justification justification) :
        base_(base), text_(std::move(text)), font_scale_(font_scale), alpha_(alpha),
        justification_(justification) {
        setInterceptsMouseClicks(false, false);
    }

    void UISettingText::paint(juce::Graphics& g) {
        const auto font_size = base_.getFontSize();
        g.setColour(base_.getTextColour().withAlpha(alpha_));
        g.setFont(juce::FontOptions{font_size * font_scale_});
        g.drawFittedText(text_, getLocalBounds().reduced(juce::roundToInt(font_size * .72f), 0),
                         justification_, 1);
    }

    UISettingPanelBackground::UISettingPanelBackground(zlgui::UIBase& base) : base_(base) {
        setInterceptsMouseClicks(false, false);
        setAlpha(preset_style::kBackgroundAlpha);
    }

    void UISettingPanelBackground::paint(juce::Graphics& g) {
        const auto font_size = base_.getFontSize();
        const auto padding = getPaddingSize(font_size);
        const auto bound = getLocalBounds().reduced(padding);
        juce::Path path;
        path.addRoundedRectangle(bound.toFloat(), static_cast<float>(padding));

        const juce::DropShadow shadow{base_.getTextColour().withAlpha(.5f), padding, {0, 0}};
        shadow.drawForPath(g, path);
        g.setColour(preset_style::surfaceColour(base_.getBackgroundColour(), base_.getTextColour()));
        g.fillPath(path);

        g.setColour(base_.getBackgroundColour());
        for (const auto& surface_bound : surface_bounds_) {
            g.fillRoundedRectangle(surface_bound.toFloat(), static_cast<float>(padding) * .75f);
        }
    }

    void UISettingPanelBackground::setSurfaceBounds(std::vector<juce::Rectangle<int>> bounds) {
        surface_bounds_ = std::move(bounds);
        repaint();
    }

    UISettingTabBar::UISettingTabBar(zlgui::UIBase& base) : base_(base) {
        setMouseCursor(juce::MouseCursor::PointingHandCursor);
        setWantsKeyboardFocus(true);
    }

    void UISettingTabBar::paint(juce::Graphics& g) {
        const auto font_size = base_.getFontSize();
        g.setFont(juce::FontOptions{preset_style::textFontSize(font_size)});

        for (auto index = 0; index < static_cast<int>(tab_names_.size()); ++index) {
            const auto selected = index == selected_index_;
            const auto hovered = index == hovered_index_;
            const auto tab_bounds = getTabBounds(index);
            const auto card = tab_bounds.toFloat().reduced(font_size * .16f);
            if (selected || hovered) {
                g.setColour(base_.getTextColour().withAlpha(selected ? .105f : .045f));
                g.fillRoundedRectangle(card, font_size * .35f);
            }

            g.setColour(base_.getTextColour().withAlpha(selected ? .95f : .62f));
            g.drawFittedText(tab_names_[static_cast<size_t>(index)],
                             tab_bounds.reduced(juce::roundToInt(font_size * .5f), 0),
                             juce::Justification::centred, 1);
        }
    }

    void UISettingTabBar::mouseMove(const juce::MouseEvent& event) {
        const auto next_index = getTabAt(event.getPosition());
        if (hovered_index_ != next_index) {
            hovered_index_ = next_index;
            repaint();
        }
    }

    void UISettingTabBar::mouseExit(const juce::MouseEvent&) {
        if (hovered_index_ >= 0) {
            hovered_index_ = -1;
            repaint();
        }
    }

    void UISettingTabBar::mouseDown(const juce::MouseEvent& event) {
        mouse_down_index_ = event.mods.isPopupMenu() ? -1 : getTabAt(event.getPosition());
        if (mouse_down_index_ >= 0) {
            grabKeyboardFocus();
        }
    }

    void UISettingTabBar::mouseUp(const juce::MouseEvent& event) {
        const auto index = getTabAt(event.getPosition());
        if (!event.mods.isPopupMenu() && event.mouseWasClicked() && index == mouse_down_index_) {
            selectTab(index, true);
        }
        mouse_down_index_ = -1;
    }

    bool UISettingTabBar::keyPressed(const juce::KeyPress& key) {
        if (key == juce::KeyPress::leftKey) {
            selectTab(juce::jmax(0, selected_index_ - 1), true);
            return true;
        }
        if (key == juce::KeyPress::rightKey) {
            selectTab(juce::jmin(static_cast<int>(tab_names_.size()) - 1, selected_index_ + 1), true);
            return true;
        }
        if (key == juce::KeyPress::homeKey) {
            selectTab(0, true);
            return true;
        }
        if (key == juce::KeyPress::endKey) {
            selectTab(static_cast<int>(tab_names_.size()) - 1, true);
            return true;
        }
        return false;
    }

    void UISettingTabBar::setSelectedIndex(const int index) {
        selectTab(index, false);
    }

    juce::Rectangle<int> UISettingTabBar::getTabBounds(const int index) const {
        if (!juce::isPositiveAndBelow(index, static_cast<int>(tab_names_.size()))) {
            return {};
        }
        const auto bounds = getLocalBounds();
        const auto left = bounds.getX() + bounds.getWidth() * index / static_cast<int>(tab_names_.size());
        const auto right = bounds.getX() + bounds.getWidth() * (index + 1) / static_cast<int>(tab_names_.size());
        return {left, bounds.getY(), right - left, bounds.getHeight()};
    }

    int UISettingTabBar::getTabAt(const juce::Point<int> position) const {
        if (!getLocalBounds().contains(position)) {
            return -1;
        }
        for (auto index = 0; index < static_cast<int>(tab_names_.size()); ++index) {
            if (getTabBounds(index).contains(position)) {
                return index;
            }
        }
        return -1;
    }

    void UISettingTabBar::selectTab(const int index, const bool send_notification) {
        if (!juce::isPositiveAndBelow(index, static_cast<int>(tab_names_.size())) ||
            selected_index_ == index) {
            return;
        }
        selected_index_ = index;
        repaint();
        if (send_notification && onTabSelected) {
            onTabSelected(index);
        }
    }

    UISettingViewport::UISettingViewport(zlgui::UIBase& base) : base_(base) {
        setWantsKeyboardFocus(true);
        setOpaque(false);
    }

    UISettingViewport::~UISettingViewport() {
        if (viewed_component_ != nullptr) {
            removeChildComponent(viewed_component_);
        }
    }

    void UISettingViewport::paint(juce::Graphics& g) {
        if (!needsScrollBar()) {
            return;
        }

        const auto font_size = base_.getFontSize();
        const auto track = getScrollTrackBounds().toFloat();
        const auto track_width = juce::jmax(1.f, font_size * .12f);
        g.setColour(base_.getTextColour().withAlpha(.055f));
        g.fillRoundedRectangle(track.withSizeKeepingCentre(track_width, track.getHeight()),
                               track_width * .5f);

        const auto thumb = getScrollThumbBounds().toFloat();
        g.setColour(base_.getTextColour().withAlpha(scroll_bar_dragging_ ? .42f
                                                       : scroll_bar_hovered_ ? .3f : .2f));
        g.fillRoundedRectangle(thumb, thumb.getWidth() * .5f);
    }

    void UISettingViewport::resized() {
        setViewPosition(view_position_);
        updateViewedComponentBounds();
    }

    void UISettingViewport::mouseMove(const juce::MouseEvent& event) {
        const auto is_hovered = needsScrollBar() && getScrollTrackBounds().contains(event.getPosition());
        if (scroll_bar_hovered_ != is_hovered) {
            scroll_bar_hovered_ = is_hovered;
            repaint(getScrollTrackBounds());
        }
    }

    void UISettingViewport::mouseExit(const juce::MouseEvent&) {
        if (scroll_bar_hovered_ && !scroll_bar_dragging_) {
            scroll_bar_hovered_ = false;
            repaint(getScrollTrackBounds());
        }
    }

    void UISettingViewport::mouseDown(const juce::MouseEvent& event) {
        if (event.mods.isPopupMenu() || !needsScrollBar() ||
            !getScrollTrackBounds().contains(event.getPosition())) {
            return;
        }

        grabKeyboardFocus();
        const auto thumb = getScrollThumbBounds();
        if (thumb.contains(event.getPosition())) {
            scroll_bar_dragging_ = true;
            drag_offset_ = event.y - thumb.getY();
        } else {
            const auto direction = event.y < thumb.getY() ? -1.0 : 1.0;
            requestViewPosition(target_view_position_ +
                                direction * static_cast<double>(getContentBounds().getHeight()) * .85);
        }
        repaint(getScrollTrackBounds());
    }

    void UISettingViewport::mouseDrag(const juce::MouseEvent& event) {
        if (!scroll_bar_dragging_) {
            return;
        }
        const auto track = getScrollTrackBounds();
        const auto thumb = getScrollThumbBounds();
        const auto travel = juce::jmax(1, track.getHeight() - thumb.getHeight());
        const auto thumb_position = juce::jlimit(0, travel, event.y - drag_offset_ - track.getY());
        requestViewPosition(getMaximumViewPosition() * static_cast<double>(thumb_position) /
                            static_cast<double>(travel));
    }

    void UISettingViewport::mouseUp(const juce::MouseEvent&) {
        if (scroll_bar_dragging_) {
            scroll_bar_dragging_ = false;
            repaint(getScrollTrackBounds());
        }
    }

    void UISettingViewport::mouseWheelMove(const juce::MouseEvent&,
                                           const juce::MouseWheelDetails& wheel) {
        if (!needsScrollBar()) {
            return;
        }
        const auto multiplier = wheel.isSmooth ? 4.5 : 3.0;
        requestViewPosition(target_view_position_ - static_cast<double>(wheel.deltaY) *
                            static_cast<double>(base_.getFontSize()) * multiplier);
    }

    bool UISettingViewport::keyPressed(const juce::KeyPress& key) {
        const auto visible_height = static_cast<double>(getContentBounds().getHeight());
        if (key == juce::KeyPress::upKey) {
            requestViewPosition(target_view_position_ - base_.getFontSize() * 2.5);
            return true;
        }
        if (key == juce::KeyPress::downKey) {
            requestViewPosition(target_view_position_ + base_.getFontSize() * 2.5);
            return true;
        }
        if (key == juce::KeyPress::pageUpKey) {
            requestViewPosition(target_view_position_ - visible_height * .85);
            return true;
        }
        if (key == juce::KeyPress::pageDownKey) {
            requestViewPosition(target_view_position_ + visible_height * .85);
            return true;
        }
        if (key == juce::KeyPress::homeKey) {
            requestViewPosition(0.0);
            return true;
        }
        if (key == juce::KeyPress::endKey) {
            requestViewPosition(getMaximumViewPosition());
            return true;
        }
        return false;
    }

    void UISettingViewport::setViewedComponent(juce::Component* component, const int content_height) {
        if (viewed_component_ != component) {
            if (viewed_component_ != nullptr) {
                removeChildComponent(viewed_component_);
            }
            viewed_component_ = component;
            view_position_ = 0.0;
            target_view_position_ = 0.0;
            scroll_update_pending_ = false;
            if (viewed_component_ != nullptr) {
                addAndMakeVisible(viewed_component_);
            }
        }
        content_height_ = juce::jmax(0, content_height);
        setViewPosition(target_view_position_);
        updateViewedComponentBounds();
        repaint();
    }

    void UISettingViewport::setViewPosition(const double position) {
        const auto next_position = juce::jlimit(0.0, getMaximumViewPosition(), position);
        target_view_position_ = next_position;
        scroll_update_pending_ = false;
        if (std::abs(next_position - view_position_) > .01) {
            view_position_ = next_position;
            updateViewedComponentBounds();
            repaint();
        } else {
            view_position_ = next_position;
        }
    }

    double UISettingViewport::getViewPosition() const {
        return target_view_position_;
    }

    void UISettingViewport::flushPendingScroll() {
        if (!scroll_update_pending_) {
            return;
        }
        scroll_update_pending_ = false;
        const auto next_position = juce::jlimit(0.0, getMaximumViewPosition(), target_view_position_);
        target_view_position_ = next_position;
        if (std::abs(next_position - view_position_) > .01) {
            view_position_ = next_position;
            updateViewedComponentBounds();
            repaint();
        }
    }

    juce::Rectangle<int> UISettingViewport::getContentBounds() const {
        const auto inset = juce::roundToInt(base_.getFontSize() * .28f);
        auto bounds = getLocalBounds().reduced(inset);
        if (needsScrollBar()) {
            bounds.removeFromRight(juce::roundToInt(base_.getFontSize() * .7f));
        }
        return bounds;
    }

    juce::Rectangle<int> UISettingViewport::getScrollTrackBounds() const {
        const auto inset = juce::roundToInt(base_.getFontSize() * .28f);
        auto bounds = getLocalBounds().reduced(inset);
        return bounds.removeFromRight(juce::roundToInt(base_.getFontSize() * .5f));
    }

    juce::Rectangle<int> UISettingViewport::getScrollThumbBounds() const {
        const auto track = getScrollTrackBounds();
        if (!needsScrollBar() || track.isEmpty()) {
            return {};
        }
        const auto visible_height = static_cast<double>(getContentBounds().getHeight());
        const auto thumb_height = juce::jlimit(
            juce::roundToInt(base_.getFontSize() * 2.f), track.getHeight(),
            juce::roundToInt(static_cast<double>(track.getHeight()) * visible_height /
                             static_cast<double>(content_height_)));
        const auto travel = juce::jmax(0, track.getHeight() - thumb_height);
        const auto maximum = getMaximumViewPosition();
        const auto thumb_y = maximum > 0.0
                                 ? juce::roundToInt(static_cast<double>(travel) * view_position_ / maximum)
                                 : 0;
        const auto thumb_width = juce::jmax(2, juce::roundToInt(base_.getFontSize() * .22f));
        return track.withWidth(thumb_width).withX(track.getCentreX() - thumb_width / 2)
                    .withY(track.getY() + thumb_y).withHeight(thumb_height);
    }

    double UISettingViewport::getMaximumViewPosition() const {
        const auto inset = juce::roundToInt(base_.getFontSize() * .28f);
        const auto visible_height = juce::jmax(0, getHeight() - 2 * inset);
        return juce::jmax(0.0, static_cast<double>(content_height_ - visible_height));
    }

    bool UISettingViewport::needsScrollBar() const {
        const auto inset = juce::roundToInt(base_.getFontSize() * .28f);
        return content_height_ > juce::jmax(0, getHeight() - 2 * inset);
    }

    void UISettingViewport::requestViewPosition(const double position) {
        const auto next_position = juce::jlimit(0.0, getMaximumViewPosition(), position);
        target_view_position_ = next_position;
        scroll_update_pending_ = std::abs(target_view_position_ - view_position_) > .01;
    }

    void UISettingViewport::updateViewedComponentBounds() {
        if (viewed_component_ == nullptr) {
            return;
        }
        const auto content_bounds = getContentBounds();
        viewed_component_->setBounds(content_bounds.getX(),
                                     content_bounds.getY() - juce::roundToInt(view_position_),
                                     content_bounds.getWidth(), content_height_);
    }
}

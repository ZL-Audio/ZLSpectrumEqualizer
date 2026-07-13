// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include "zlp/zlp.hpp"
#include "state/state.hpp"

class PluginProcessor final : public juce::AudioProcessor {
public:
    zlstate::DummyProcessor dummy_processor_;
    juce::AudioProcessorValueTreeState parameters_;
    juce::AudioProcessorValueTreeState parameters_NA_;

    PluginProcessor();

    ~PluginProcessor() override;

    void prepareToPlay(double sample_rate, int samples_per_block) override;

    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    void processBlockBypassed(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;

    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;

    bool producesMidi() const override;

    bool isMidiEffect() const override;

    double getTailLengthSeconds() const override;

    int getNumPrograms() override;

    int getCurrentProgram() override;

    void setCurrentProgram(int index) override;

    const juce::String getProgramName(int index) override;

    void changeProgramName(int index, const juce::String& new_name) override;

    void getStateInformation(juce::MemoryBlock& dest_data) override;

    void setStateInformation(const void* data, int size_in_bytes) override;

    bool supportsDoublePrecisionProcessing() const override { return false; }

    zlp::Controller& getController() {
        return controller_;
    }

    double getAtomicSampleRate() const {
        return sample_rate_.load(std::memory_order::relaxed);
    }

private:
    zlp::Controller controller_;
    std::array<std::unique_ptr<zlp::FilterAttach>, zlp::kBandNum> filter_attachments_;
    std::array<std::unique_ptr<zlp::FilterDynamicAttach>, zlp::kBandNum> filter_dynamic_attachments_;

    std::atomic<double> sample_rate_{48000.0};
    std::atomic<float>& a_bypass_;

    std::vector<float> dummy_main_;
    std::vector<float> dummy_side_;
    std::array<float*, 4> pointers_{nullptr};

    enum ChannelLayout {
        kMain1Aux0, kMain1Aux1, kMain1Aux2,
        kMain2Aux0, kMain2Aux1, kMain2Aux2,
        kInvalid
    };

    ChannelLayout channel_layout_{kInvalid};

    bool update_channel_layout_per_call_{false};

    void updateChannelLayout();

    void processBlockInternal(juce::AudioBuffer<float>& buffer, bool bypass);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};

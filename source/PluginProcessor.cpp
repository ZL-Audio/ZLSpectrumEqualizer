// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "PluginProcessor.h"

#include <numbers>

#include "PluginEditor.h"

//==============================================================================
PluginProcessor::PluginProcessor() :
    AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withInput("Aux", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)
        ),
    dummy_processor_(),
    parameters_(*this, nullptr,
                juce::Identifier("ZLSpectrumEqualizerParameters"),
                zlp::getParameterLayout()),
    parameters_NA_(dummy_processor_, nullptr,
                   juce::Identifier("ZLSpectrumEqualizerNAParameters"),
                   zlstate::getNAParameterLayout()),
    controller_(*this),
    a_bypass_(*parameters_.getRawParameterValue(zlp::PBypass::kID)) {
}

PluginProcessor::~PluginProcessor() = default;

const juce::String PluginProcessor::getName() const {
    return JucePlugin_Name;
}

bool PluginProcessor::acceptsMidi() const {
    return false;
}

bool PluginProcessor::producesMidi() const {
    return false;
}

bool PluginProcessor::isMidiEffect() const {
    return false;
}

double PluginProcessor::getTailLengthSeconds() const {
    return 0.0;
}

int PluginProcessor::getNumPrograms() {
    return 1;
}

int PluginProcessor::getCurrentProgram() {
    return 0;
}

void PluginProcessor::setCurrentProgram(int) {
}

const juce::String PluginProcessor::getProgramName(int) {
    return "Default";
}

void PluginProcessor::changeProgramName(int, const juce::String&) {
}

void PluginProcessor::prepareToPlay(const double sample_rate, const int samples_per_block) {
    dummy_main_.resize(static_cast<size_t>(samples_per_block));
    dummy_side_.resize(static_cast<size_t>(samples_per_block));
    const juce::PluginHostType hostType;
    update_channel_layout_per_call_ = hostType.isMaschine();

    controller_.prepare(sample_rate, static_cast<size_t>(samples_per_block));

    updateChannelLayout();
}

void PluginProcessor::releaseResources() {
}

bool PluginProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    if (layouts.getMainInputChannelSet() == juce::AudioChannelSet::stereo() &&
        layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo() &&
        (layouts.getChannelSet(true, 1).isDisabled() ||
            layouts.getChannelSet(true, 1) == juce::AudioChannelSet::mono() ||
            layouts.getChannelSet(true, 1) == juce::AudioChannelSet::stereo())) {
        return true;
    }
    if (layouts.getMainInputChannelSet() == juce::AudioChannelSet::mono() &&
        layouts.getMainOutputChannelSet() == juce::AudioChannelSet::mono() &&
        (layouts.getChannelSet(true, 1).isDisabled() ||
            layouts.getChannelSet(true, 1) == juce::AudioChannelSet::mono() ||
            layouts.getChannelSet(true, 1) == juce::AudioChannelSet::stereo())) {
        return true;
    }
    return false;
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                   juce::MidiBuffer&) {

    const auto bypass = a_bypass_.load(std::memory_order::relaxed) > 5.f;
    processBlockInternal(buffer, bypass);
}

void PluginProcessor::processBlockBypassed(juce::AudioBuffer<float>& buffer,
                                           juce::MidiBuffer&) {
    processBlockInternal(buffer, true);
}


bool PluginProcessor::hasEditor() const {
    return true;
}

juce::AudioProcessorEditor* PluginProcessor::createEditor() {
    // return new PluginEditor(*this);
    return new juce::GenericAudioProcessorEditor(*this);
}

void PluginProcessor::getStateInformation(juce::MemoryBlock& dest_data) {
    auto temp_tree = juce::ValueTree("ZLSpectrumEqualizerParaState");
    temp_tree.appendChild(parameters_.copyState(), nullptr);
    temp_tree.appendChild(parameters_NA_.copyState(), nullptr);
    const std::unique_ptr<juce::XmlElement> xml(temp_tree.createXml());
    copyXmlToBinary(*xml, dest_data);
}

void PluginProcessor::setStateInformation(const void* data, const int size_in_bytes) {
    std::unique_ptr<juce::XmlElement> xml_state(getXmlFromBinary(data, size_in_bytes));
    if (xml_state != nullptr && xml_state->hasTagName("ZLSpectrumEqualizerParaState")) {
        const auto temp_tree = juce::ValueTree::fromXml(*xml_state);
        parameters_.replaceState(temp_tree.getChildWithName(parameters_.state.getType()));
        parameters_NA_.replaceState(temp_tree.getChildWithName(parameters_NA_.state.getType()));
    }
}

void PluginProcessor::updateChannelLayout() {
    const auto* main_bus = getBus(true, 0);
    const auto* aux_bus = getBus(true, 1);
    channel_layout_ = kInvalid;
    if (main_bus == nullptr) {
        return;
    }
    if (main_bus->getCurrentLayout() == juce::AudioChannelSet::mono()) {
        if (aux_bus == nullptr || !aux_bus->isEnabled()) {
            channel_layout_ = kMain1Aux0;
        } else if (aux_bus->getCurrentLayout() == juce::AudioChannelSet::mono()) {
            channel_layout_ = kMain1Aux1;
        } else if (aux_bus->getCurrentLayout() == juce::AudioChannelSet::stereo()) {
            channel_layout_ = kMain1Aux2;
        }
    } else if (main_bus->getCurrentLayout() == juce::AudioChannelSet::stereo()) {
        if (aux_bus == nullptr || !aux_bus->isEnabled()) {
            channel_layout_ = kMain2Aux0;
        } else if (aux_bus->getCurrentLayout() == juce::AudioChannelSet::mono()) {
            channel_layout_ = kMain2Aux1;
        } else if (aux_bus->getCurrentLayout() == juce::AudioChannelSet::stereo()) {
            channel_layout_ = kMain2Aux2;
        }
    }
    switch (channel_layout_) {
    case kMain1Aux0: {
        pointers_[1] = dummy_main_.data();
        pointers_[2] = nullptr;
        pointers_[3] = nullptr;
        break;
    }
    case kMain1Aux1: {
        pointers_[1] = dummy_main_.data();
        pointers_[3] = dummy_side_.data();
        break;
    }
    case kMain1Aux2: {
        pointers_[1] = dummy_main_.data();
        break;
    }
    case kMain2Aux0: {
        pointers_[2] = nullptr;
        pointers_[3] = nullptr;
        break;
    }
    case kMain2Aux1: {
        pointers_[3] = dummy_side_.data();
        break;
    }
    case kMain2Aux2: {
        break;
    }
    case kInvalid: {
        return;
    }
    }
}

void PluginProcessor::processBlockInternal(juce::AudioBuffer<float>& buffer, const bool bypass) {
    juce::ScopedNoDenormals noDenormals;
    if (buffer.getNumSamples() == 0) {
        return; // ignore empty blocks
    }
    controller_.prepareBuffer();
    const auto ext_side = controller_.getExtSide();
    const auto num_samples = static_cast<size_t>(buffer.getNumSamples());
    switch (channel_layout_) {
    case kMain1Aux0: {
        pointers_[0] = buffer.getWritePointer(0);
        zldsp::vector::copy(pointers_[1], pointers_[0], num_samples);

        controller_.process(pointers_, num_samples, bypass);
        break;
    }
    case kMain1Aux1: {
        pointers_[0] = buffer.getWritePointer(0);
        zldsp::vector::copy(pointers_[1], pointers_[0], num_samples);
        if (ext_side) {
            pointers_[2] = buffer.getWritePointer(1);
            zldsp::vector::copy(pointers_[3], pointers_[2], num_samples);
        }

        controller_.process(pointers_, num_samples, bypass);
        break;
    }
    case kMain1Aux2: {
        pointers_[0] = buffer.getWritePointer(0);
        zldsp::vector::copy(pointers_[1], pointers_[0], num_samples);
        if (ext_side) {
            pointers_[2] = buffer.getWritePointer(1);
            pointers_[3] = buffer.getWritePointer(2);
        }

        controller_.process(pointers_, num_samples, bypass);
        break;
    }
    case kMain2Aux0: {
        pointers_[0] = buffer.getWritePointer(0);
        pointers_[1] = buffer.getWritePointer(1);

        controller_.process(pointers_, num_samples, bypass);
        break;
    }
    case kMain2Aux1: {
        pointers_[0] = buffer.getWritePointer(0);
        pointers_[1] = buffer.getWritePointer(1);
        if (ext_side) {
            pointers_[2] = buffer.getWritePointer(2);
            zldsp::vector::copy(pointers_[3], pointers_[2], num_samples);
        }

        controller_.process(pointers_, num_samples, bypass);
        break;
    }
    case kMain2Aux2: {
        pointers_[0] = buffer.getWritePointer(0);
        pointers_[1] = buffer.getWritePointer(1);
        if (ext_side) {
            pointers_[2] = buffer.getWritePointer(2);
            pointers_[3] = buffer.getWritePointer(3);
        }

        controller_.process(pointers_, num_samples, bypass);
        break;
    }
    case kInvalid: {
        return;
    }
    }
}

juce::AudioProcessor*JUCE_CALLTYPE

createPluginFilter() {
    return new PluginProcessor();
}

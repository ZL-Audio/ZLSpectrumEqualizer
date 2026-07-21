// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

#include "chore_attach.hpp"

namespace zlp {
    ChoreAttach::ChoreAttach(juce::AudioProcessor&,
                             juce::AudioProcessorValueTreeState& parameters,
                             Controller& controller) :
        parameters_(parameters),
        controller_(controller) {
        for (const auto& id : kIDs) {
            parameters_.addParameterListener(id, this);
            parameterChanged(id, parameters.getRawParameterValue(id)->load(std::memory_order::relaxed));
        }
    }

    ChoreAttach::~ChoreAttach() {
        for (const auto& id : kIDs) {
            parameters_.removeParameterListener(id, this);
        }
    }

    void ChoreAttach::parameterChanged(const juce::String& parameter_ID, const float value) {
        if (parameter_ID == PExtSide::kID) {
            controller_.setExtSide(value > .5f);
        } else if (parameter_ID == PFFTResolution::kID) {
            controller_.setFFTResolution(static_cast<FFTResolution>(std::round(value)));
        } else if (parameter_ID == PSpecSmooth::kID) {
            controller_.setSpecSmoothValue(value);
        } else if (parameter_ID == PSpecSmoothType::kID) {
            controller_.setSpecSmoothType(
                static_cast<zldsp::filter::SpecSmoother<float>::SmoothMethod>(std::round(value)));
        } else if (parameter_ID == PSpecTilt::kID) {
            controller_.setSpecTiltSlope(value);
        } else if (parameter_ID == PSpecSkewAttack::kID) {
            controller_.setSpecFollowerSkewAttack(value);
        } else if (parameter_ID == PSpecSkewRelease::kID) {
            controller_.setSpecFollowerSkewRelease(value);
        } else if (parameter_ID == POutputGain::kID) {
            controller_.setOutputGain(value);
        } else if (parameter_ID == PStaticGain::kID) {
            controller_.setSGCON(value);
        }
    }
}

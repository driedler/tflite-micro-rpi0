/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/micro/python/interpreter_wrapper/model_builder.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <utility>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"


namespace tflite {

namespace {

// Ensure that ErrorReporter is non-null.
ErrorReporter* ValidateErrorReporter(ErrorReporter* e) {
  return e;
}

}  // namespace

#ifndef TFLITE_MCU
// Loads a model from `filename`. If `mmap_file` is true then use mmap,
// otherwise make a copy of the model in a buffer.
std::unique_ptr<Allocation> GetAllocationFromFile(
    const char* filename, ErrorReporter* error_reporter) {
  std::unique_ptr<Allocation> allocation;
  allocation.reset(new FileCopyAllocation(filename, error_reporter));
  return allocation;
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromFile(
    const char* filename, ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);
  return BuildFromAllocation(GetAllocationFromFile(filename, error_reporter),
                             error_reporter);
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromFile(
    const char* filename,
    ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);
  return VerifyAndBuildFromAllocation(
      GetAllocationFromFile(filename, error_reporter),
      error_reporter);
}
#endif

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromBuffer(
    const char* caller_owned_buffer, size_t buffer_size,
    ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);
  std::unique_ptr<Allocation> allocation(
      new MemoryAllocation(caller_owned_buffer, buffer_size, error_reporter));
  return BuildFromAllocation(std::move(allocation), error_reporter);
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromBuffer(
    const char* caller_owned_buffer, size_t buffer_size,
    ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);
  std::unique_ptr<Allocation> allocation(
      new MemoryAllocation(caller_owned_buffer, buffer_size, error_reporter));
  return VerifyAndBuildFromAllocation(std::move(allocation),
                                      error_reporter);
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromAllocation(
    std::unique_ptr<Allocation> allocation, ErrorReporter* error_reporter) {
  std::unique_ptr<FlatBufferModel> model(new FlatBufferModel(
      std::move(allocation), ValidateErrorReporter(error_reporter)));
  if (!model->initialized()) {
    model.reset();
  }
  return model;
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::VerifyAndBuildFromAllocation(
    std::unique_ptr<Allocation> allocation,
    ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);
  if (!allocation || !allocation->valid()) {
    TF_LITE_REPORT_ERROR(error_reporter, "The model allocation is null/empty");
    return nullptr;
  }

  flatbuffers::Verifier base_verifier(
      reinterpret_cast<const uint8_t*>(allocation->base()),
      allocation->bytes());
  if (!VerifyModelBuffer(base_verifier)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "The model is not a valid Flatbuffer buffer");
    return nullptr;
  }

  return BuildFromAllocation(std::move(allocation), error_reporter);
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromModel(
    const tflite::Model* caller_owned_model_spec,
    ErrorReporter* error_reporter) {
  error_reporter = ValidateErrorReporter(error_reporter);

  std::unique_ptr<FlatBufferModel> model(
      new FlatBufferModel(caller_owned_model_spec, error_reporter));
  if (!model->initialized()) {
    model.reset();
  }
  return model;
}


bool FlatBufferModel::CheckModelIdentifier() const {
  if (!tflite::ModelBufferHasIdentifier(allocation_->base())) {
    const char* ident = flatbuffers::GetBufferIdentifier(allocation_->base());
    error_reporter_->Report(
        "Model provided has model identifier '%c%c%c%c', should be '%s'\n",
        ident[0], ident[1], ident[2], ident[3], tflite::ModelIdentifier());
    return false;
  }
  return true;
}

FlatBufferModel::FlatBufferModel(const Model* model,
                                 ErrorReporter* error_reporter)
    : model_(model), error_reporter_(ValidateErrorReporter(error_reporter)) {}

FlatBufferModel::FlatBufferModel(std::unique_ptr<Allocation> allocation,
                                 ErrorReporter* error_reporter)
    : error_reporter_(ValidateErrorReporter(error_reporter)),
      allocation_(std::move(allocation)) {
  if (!allocation_ || !allocation_->valid() || !CheckModelIdentifier()) {
    return;
  }

  model_ = ::tflite::GetModel(allocation_->base());
}

FlatBufferModel::~FlatBufferModel() {}

}  // namespace tflite

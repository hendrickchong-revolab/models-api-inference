# SPDX-License-Identifier: Apache-2.0
import re
import time
import math
import asyncio

from collections import Counter
from functools import cached_property
from typing import List, Optional, Tuple, Union, Iterator

import numpy as np
import torch

from fastapi import Request
from silero_vad import load_silero_vad
from torchaudio.io import StreamReader
from torio.io._streaming_media_decoder import ChunkTensor

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
	ErrorResponse,
	TranscriptionResponse,
	TranscriptionResponseVerbose,
	TranscriptionSegment,
)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger

from app.protocol import TranscriptionRequestImprove

silero = load_silero_vad(onnx=True)
frame_size = 512
buffer_size = 4096
sample_rate = 16000
segment_length = sample_rate
maxlen = 30
pattern_pair = r"<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>"

logger = init_logger(__name__)


class OpenAIServingTranscription(OpenAIServing):
	def __init__(
		self,
		engine_client: EngineClient,
		model_config: ModelConfig,
		models: OpenAIServingModels,
		*,
		request_logger: Optional[RequestLogger],
		return_tokens_as_token_ids: bool = False,
	):
		super().__init__(
			engine_client=engine_client,
			model_config=model_config,
			models=models,
			request_logger=request_logger,
			return_tokens_as_token_ids=return_tokens_as_token_ids,
		)

		diff_sampling_param = self.model_config.get_diff_sampling_param()
		if diff_sampling_param:
			logger.info(
				"Overwriting default completion sampling param with: %s",
				diff_sampling_param,
			)
		self.tokenizer = None

	async def _completion(self, prompt, sampling_params, request_id):
		result_generator = self.engine_client.generate(
			prompt,
			sampling_params,
			request_id,
		)
		try:
			assert result_generator is not None
			async for op in result_generator:
				result = op
			return result
		except asyncio.CancelledError:
			return self.create_error_response("Client disconnected")
		except ValueError as e:
			return self.create_error_response(str(e))

	@cached_property
	def language_tokens(self):
		non_lang_special_tokens = [
			"<|translate|>",
			"<|transcribe|>",
			"<|startoflm|>",
			"<|startofprev|>",
			"<|nospeech|>",
			"<|notimestamps|>",
			"<|startoftranscript|>",
		]
		language_tokens = [
			t for t in self.tokenizer.additional_special_tokens if t not in non_lang_special_tokens
		]
		return language_tokens

	async def _transcription(
		self,
		request: TranscriptionRequestImprove,
		y,
		request_id,
	):
		start_ts = time.perf_counter()
		if self.tokenizer is None:
			self.tokenizer = await self.engine_client.get_tokenizer()

		audio = (y, sample_rate)

		lang_token = None
		if request.language:
			logger.info(f"Receiving external request language of {request.language}")
			lang_token = f"<|{request.language}|>"
			if lang_token not in self.language_tokens:
				logger.warning(
					f"Lang token of {lang_token} isn't available on list of valid lang tokens. Ignoring the language params!"
				)
				lang_token = None

		if not lang_token:
			prompt = {
				"encoder_prompt": {
					"prompt": "",
					"multi_modal_data": {
						"audio": audio,
					},
				},
				"decoder_prompt": "<|startoftranscript|>",
			}
			default_params = self.model_config.get_diff_sampling_param()
			sampling_params = request.to_sampling_params(1, default_params)
			result = await self._completion(prompt, sampling_params, request_id + "_lang")
			if isinstance(result, ErrorResponse):
				return result
			lang_token = self.tokenizer.convert_ids_to_tokens(
				result.outputs[0].token_ids
			)[0]

			if lang_token in ("<|ar|>", "<|lev|>"):
				lang_token = "<|lev|>"
			else:
				lang_token = "<|en|>"

		decoder_prompt = f"<|startoftranscript|>{lang_token}<|transcribe|><|0.00|>"
		prompt = {
			"encoder_prompt": {
				"prompt": "",
				"multi_modal_data": {
					"audio": audio,
				},
			},
			"decoder_prompt": decoder_prompt,
		}
		default_max_tokens = self.model_config.max_model_len
		default_params = self.model_config.get_diff_sampling_param()
		sampling_params = request.to_sampling_params(default_max_tokens, default_params)

		self._log_inputs(
			request_id,
			prompt["decoder_prompt"],
			params=sampling_params,
			lora_request=None,
		)

		start_model_gen_ts = time.perf_counter()
		result = await self._completion(prompt, sampling_params, request_id)
		end_model_gen_ts = time.perf_counter()
		if isinstance(result, ErrorResponse):
			return result
		tokens = self.tokenizer.convert_tokens_to_string(
			self.tokenizer.convert_ids_to_tokens(result.outputs[0].token_ids)
		)
		end_ts = time.perf_counter()
		return (
			f"{decoder_prompt}{tokens}",
			(end_ts - start_ts) * 1000,
			(end_model_gen_ts - start_model_gen_ts) * 1000,
		)

	@staticmethod
	def calculate_rtf(perf_data):
		audio_dur = perf_data.get("audio_dur")
		if audio_dur:
			audio_dur_ms = audio_dur * 1000
			perf_data["stt_model_gen_rtf"] = perf_data.get("stt_model_gen_latency_ms", 0) / audio_dur_ms
			perf_data["transcription_rtf"] = perf_data.get("transcription_latency_ms", 0) / audio_dur_ms
			if "vad_latency_ms" in perf_data.keys():
				perf_data["vad_rtf"] = perf_data.get("vad_latency_ms", 0) / audio_dur_ms
		return perf_data

	async def create_transcription(
		self, audio_data, request: TranscriptionRequestImprove, raw_request: Request
	) -> Union[TranscriptionResponse, TranscriptionResponseVerbose, ErrorResponse]:
		"""Transcription API similar to OpenAI's API.

		See https://platform.openai.com/docs/api-reference/audio/createTranscription
		for the API specification. This API mimics the OpenAI transcription API.
		"""
		error_check_ret = await self._check_model(request)
		if error_check_ret is not None:
			return error_check_ret

		if self.engine_client.errored:
			raise self.engine_client.dead_error

		request_id = f"cmpl-{self._base_request_id(raw_request)}"
		if request.client_reference_id:
			request_id += ("-client-" + request.client_reference_id)

		start_vad_process_ts = time.perf_counter()
		vad_data_chunks, (raw_audio, raw_sr) = self._chunk_audio(audio_data, request)
		end_vad_process_ts = time.perf_counter()

		if not vad_data_chunks:
			no_segment_msg = "No valid Speech Segments found! Please check if your Audio contains any speech."
			return ErrorResponse(message=no_segment_msg, type="error", code=400)

		results = await self._transcribe_chunks(request, vad_data_chunks, request_id)

		for result in results:
			if isinstance(result, ErrorResponse):
				return result

		segments, full_text, language, perf_tracker = self._postprocess_result(
			results, vad_data_chunks, request
		)
		perf_tracker["vad_latency_ms"] = (end_vad_process_ts - start_vad_process_ts) * 1000
		perf_tracker.pop("chunk_perf_tracker", []*len(segments))
		usage = {
			"type": "duration",
			"seconds": int(round(float(perf_tracker.get("audio_dur", 0)))),
		}

		if request.response_format == "verbose_json":
			return self._build_verbose_response(segments, full_text, language, usage)
		elif request.response_format == "json":
			return TranscriptionResponse(text=full_text, usage=usage)
		return full_text

	@staticmethod
	def _construct_streamer(audio_data, frames_per_chunk) -> Iterator[Tuple[Optional[ChunkTensor], ...]]:
		streamer = StreamReader(
			src=audio_data, format=None, option=None, buffer_size=buffer_size
		)

		streamer.add_basic_audio_stream(
			stream_index=0,
			sample_rate=None,
			buffer_chunk_size=-1,
			frames_per_chunk=math.ceil(
				streamer.get_src_stream_info(0).sample_rate / sample_rate * frames_per_chunk
			),
		)
		streamer.add_basic_audio_stream(
			stream_index=0,
			sample_rate=sample_rate,
			buffer_chunk_size=-1,
			frames_per_chunk=frames_per_chunk,
		)

		return streamer

	def _chunk_audio(
		self, audio_data, request
	) -> List[Tuple[np.ndarray, float, float]]:
		frames_per_chunk = segment_length if request.chunking_method == "naive" else frame_size

		streamer = self._construct_streamer(audio_data, frames_per_chunk)

		chunks_final, raw_audio_arr = [], []
		subchunk_frames = []
		last_timestamp = total_audio = total_silent = total_silent_frames = total_frames = 0

		for chunks in streamer.stream():
			orig_frame, frame = chunks
			frame = frame.to(torch.float32)

			raw_audio_arr.append(orig_frame.to(torch.float32))

			if frame.ndim == 2:
				frame = frame.mean(dim=-1)
			elif not 1 <= frame.ndim <= 2:
				raise ValueError("Frame ndims must be either 1 or 2!")

			total_frames += 1

			total_audio += len(frame)
			audio_len_s = total_audio / sample_rate
			subchunk_frames.append(frame.numpy())

			if request.chunking_method == "vad":
				if len(frame) < frames_per_chunk:
					continue
				is_voice = silero(frame, sr=sample_rate).numpy()[0][0] > request.vad_confidence
				if is_voice:
					total_silent = 0
				else:
					total_silent += len(frame)
					total_silent_frames += 1

			last_prev_chunk = frame

			negative_ratio = (total_silent_frames / total_frames if total_frames > 0 else 0)
			silent_len_ms = (total_silent / sample_rate) * 1000
			vad_trigger = self._should_trigger_chunk(audio_len_s, silent_len_ms, request)

			if vad_trigger or audio_len_s >= maxlen:
				if negative_ratio <= request.reject_segment_vad_ratio:
					wav_data = np.concatenate(subchunk_frames)
					chunks_final.append(
						(wav_data, last_timestamp, last_timestamp + audio_len_s)
					)
				last_timestamp += audio_len_s
				subchunk_frames = []
				total_audio = total_silent = total_silent_frames = total_frames = 0

		if len(subchunk_frames):
			if request.chunking_method == "vad":
				combined_last_chunk = torch.cat([last_prev_chunk, frame], dim=0)[-frames_per_chunk:]
				is_voice = silero(combined_last_chunk, sr=sample_rate).numpy()[0][0] > request.vad_confidence
				del combined_last_chunk
				if is_voice:
					total_silent = 0
				else:
					total_silent += len(frame)
					total_silent_frames += 1

			negative_ratio = (
				total_silent_frames / total_frames if total_frames > 0 else 0
			)
			wav_data = np.concatenate(subchunk_frames)
			if negative_ratio <= request.reject_segment_vad_ratio:
				chunks_final.append((wav_data, last_timestamp, last_timestamp + audio_len_s))

		return chunks_final, (
			torch.cat(raw_audio_arr, dim=0).numpy(),
			int(streamer.get_src_stream_info(0).sample_rate),
		)

	def _should_trigger_chunk(self, audio_len_s, silent_len, request):
		return (
			audio_len_s * 1000 >= request.minimum_trigger_vad_ms
			and silent_len >= request.minimum_silent_ms
		)

	async def _transcribe_chunks(self, request, chunks, request_id):
		tasks = [
			asyncio.create_task(
				self._transcription(request, chunk[0], f"{request_id}_{no}")
			)
			for no, chunk in enumerate(chunks)
		]

		return await asyncio.gather(*tasks)

	def __clean_transcription(self, text: str):
		tokenized_ids = self.tokenizer(text, add_special_tokens=False).input_ids
		return self.tokenizer.decode(tokenized_ids, skip_special_tokens=True).strip() + " "

	def _agg_language(self, languages: List = []):
		priority = {"lev": 3, "en": 2, "unknown": 1}

		if not languages:
			return "unknown"

		counts = Counter(languages)
		return max(counts.items(), key=lambda x: (x[1], priority.get(x[0], 0)))[0]

	def _extract_language(self, tokens):
		try:
			lang = tokens.split("<|")[2].split("|>")[0]
		except (AttributeError, TypeError, IndexError) as e:
			logger.debug(f"Failed to extract segment lang due to {e}")
			lang = "unknown"

		return lang

	def _postprocess_result(self, results, chunks, request):
		segments = []
		all_texts = []
		segments_language = []

		perf_tracker = {
			"stt_model_gen_latency_ms": 0,
			"transcription_latency_ms": 0,
			"audio_dur": 0,
			"chunk_perf_tracker": [],
		}

		segment_no = 0
		for vad_no, (result, transcription_latency, stt_model_gen_latency) in enumerate(results):
			vad_start_ts, vad_end_ts = chunks[vad_no][1], chunks[vad_no][2]

			vad_audio_dur = vad_end_ts - vad_start_ts

			perf_tracker["audio_dur"] += vad_audio_dur
			perf_tracker["transcription_latency_ms"] += transcription_latency
			perf_tracker["stt_model_gen_latency_ms"] += stt_model_gen_latency

			lang = self._extract_language(result)
			segments_language.append(lang)

			matches = re.findall(pattern_pair, result)
			if matches:
				for m in matches:
					text = m[1].strip()
					text = self.__clean_transcription(text)
					segment = self._build_segment(
						segment_no,
						vad_no,
						text,
						float(m[0]) + vad_start_ts,
						float(m[2]) + vad_start_ts,
						lang,
						request,
					)
					segments.append(segment)
					all_texts.append(text)
					segment_no += 1

					whisper_segment_dur = float(m[2]) - float(m[0])
					perf_tracker["chunk_perf_tracker"].append(
						{
							"audio_dur": whisper_segment_dur,
							"transcription_latency_ms": transcription_latency * (whisper_segment_dur / vad_audio_dur),
							"stt_model_gen_latency_ms": stt_model_gen_latency * (whisper_segment_dur / vad_audio_dur),
							"chunk_source": "whisper-inner-chunk",
						}
					)
			else:
				result = self.__clean_transcription(result)
				segment = self._build_segment(
					segment_no,
					vad_no,
					result,
					vad_start_ts,
					vad_end_ts,
					lang,
					request,
				)
				segments.append(segment)
				all_texts.append(result)
				segment_no += 1

				perf_tracker["chunk_perf_tracker"].append(
					{
						"audio_dur": vad_audio_dur,
						"transcription_latency_ms": transcription_latency,
						"stt_model_gen_latency_ms": stt_model_gen_latency,
						"chunk_source": "vad-chunk",
					}
				)

		full_text = "".join(all_texts).strip()
		language = self._agg_language(segments_language)

		assert len(perf_tracker["chunk_perf_tracker"]) == len(segments), "Mismatch on Perf-Tracker and Segments len!"

		return segments, full_text, language, perf_tracker

	def _build_segment(self, idx, vad_idx, text, start, end, language, request):
		return TranscriptionSegment(
			id=idx,
			vad_id=vad_idx,
			avg_logprob=0.0,
			compression_ratio=0.0,
			start=start,
			end=end,
			no_speech_prob=0.0,
			seek=0,
			temperature=request.temperature,
			text=text,
			language=language,
			tokens=self.tokenizer.encode(text, add_special_tokens=False),
		)

	def _build_verbose_response(self, segments, text, language, usage):
		return TranscriptionResponseVerbose(
			duration=str(segments[-1].end),
			language=language,
			text=text,
			segments=segments,
			words=None,
			task="transcribe",
			usage=usage,
		)

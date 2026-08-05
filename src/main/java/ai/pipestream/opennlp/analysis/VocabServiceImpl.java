/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.pipestream.opennlp.analysis;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.pipestream.opennlp.analysis.v1.GetVocabDriftRequest;
import ai.pipestream.opennlp.analysis.v1.GetVocabDriftResponse;
import ai.pipestream.opennlp.analysis.v1.GetVocabStatsRequest;
import ai.pipestream.opennlp.analysis.v1.GetVocabStatsResponse;
import ai.pipestream.opennlp.analysis.v1.SnapshotVocabRequest;
import ai.pipestream.opennlp.analysis.v1.SnapshotVocabResponse;
import ai.pipestream.opennlp.analysis.v1.VocabChannelDrift;
import ai.pipestream.opennlp.analysis.v1.VocabSnapshotDescriptor;
import ai.pipestream.opennlp.analysis.v1.VocabularyServiceGrpc;
import ai.pipestream.opennlp.analysis.vocab.DriftMetrics;
import ai.pipestream.opennlp.analysis.vocab.VocabularyListener;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;

/**
 * gRPC front end of the vocabulary listener: listener state and snapshot
 * history, explicit snapshots, and drift metrics between two windows.
 * Registered only when the listener is enabled — on a listener-less server
 * every method is UNIMPLEMENTED and GetCapabilities says why.
 *
 * <p>Errors on the wire are clean gRPC statuses: INVALID_ARGUMENT for an
 * unknown snapshot reference, INTERNAL for everything else.</p>
 */
public final class VocabServiceImpl extends VocabularyServiceGrpc.VocabularyServiceImplBase {

  private static final Logger LOG = LoggerFactory.getLogger(VocabServiceImpl.class);

  private final VocabularyListener listener;

  /**
   * @param listener the vocabulary listener, must not be {@code null}
   */
  public VocabServiceImpl(VocabularyListener listener) {
    if (listener == null) {
      throw new IllegalArgumentException("listener must not be null");
    }
    this.listener = listener;
  }

  @Override
  public void getVocabStats(GetVocabStatsRequest request,
                            StreamObserver<GetVocabStatsResponse> observer) {
    try {
      observer.onNext(GetVocabStatsResponse.newBuilder()
          .setEnabled(true)
          .setWindowDocuments(listener.windowDocuments())
          .addAllLiveChannels(listener.liveStats())
          .addAllSnapshots(listener.snapshots())
          .build());
      observer.onCompleted();
    } catch (RuntimeException e) {
      LOG.error("GetVocabStats failed", e);
      observer.onError(Status.INTERNAL
          .withDescription("vocab stats failed: " + e.getMessage()).asRuntimeException());
    }
  }

  @Override
  public void snapshotVocab(SnapshotVocabRequest request,
                            StreamObserver<SnapshotVocabResponse> observer) {
    try {
      final VocabSnapshotDescriptor descriptor = listener.snapshotNow();
      final SnapshotVocabResponse.Builder response = SnapshotVocabResponse.newBuilder()
          .setPersisted(descriptor != null);
      if (descriptor != null) {
        response.setSnapshot(descriptor);
      }
      observer.onNext(response.build());
      observer.onCompleted();
    } catch (RuntimeException e) {
      LOG.error("SnapshotVocab failed", e);
      observer.onError(Status.INTERNAL
          .withDescription("vocab snapshot failed: " + e.getMessage())
          .asRuntimeException());
    }
  }

  @Override
  public void getVocabDrift(GetVocabDriftRequest request,
                            StreamObserver<GetVocabDriftResponse> observer) {
    try {
      final GetVocabDriftResponse.Builder response = GetVocabDriftResponse.newBuilder();
      for (VocabularyListener.ChannelDrift drift
          : listener.drift(request.getFrom(), request.getTo())) {
        final DriftMetrics.Result metrics = drift.metrics();
        response.addChannels(VocabChannelDrift.newBuilder()
            .setChannel(drift.channel())
            .setFromCardinality(metrics.fromCardinality())
            .setToCardinality(metrics.toCardinality())
            .setUnionCardinality(metrics.unionCardinality())
            .setNoveltyRate(metrics.noveltyRate())
            .setJensenShannonDivergence(metrics.jensenShannonDivergence())
            .setEmbeddingOovComputed(metrics.embeddingOovComputed())
            .setEmbeddingOovShare(metrics.embeddingOovShare()));
      }
      observer.onNext(response.build());
      observer.onCompleted();
    } catch (IllegalArgumentException e) {
      observer.onError(Status.INVALID_ARGUMENT
          .withDescription(e.getMessage()).asRuntimeException());
    } catch (RuntimeException e) {
      LOG.error("GetVocabDrift failed", e);
      observer.onError(Status.INTERNAL
          .withDescription("vocab drift failed: " + e.getMessage()).asRuntimeException());
    }
  }
}

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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import ai.pipestream.opennlp.analysis.pipeline.PipelineEnvironment;
import ai.pipestream.opennlp.analysis.v1.AnalysisOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisServiceGrpc;
import ai.pipestream.opennlp.analysis.v1.AnalyzeRequest;
import ai.pipestream.opennlp.analysis.v1.AnalyzeResponse;
import ai.pipestream.opennlp.analysis.v1.AnalyzeStreamDoc;
import ai.pipestream.opennlp.analysis.v1.AnalyzeStreamRequest;
import ai.pipestream.opennlp.analysis.v1.AnalyzeStreamResponse;
import ai.pipestream.opennlp.analysis.v1.TermVectorOptions;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;

/**
 * AnalyzeStream over a real Netty server and channel, the wiring
 * {@link AnalysisServer} uses, so the manual flow control runs against real
 * HTTP/2 transport windows.
 */
class AnalyzeStreamTest {

  // Small enough to exercise the oversize path with a test-sized payload.
  private static final int MAX_TEXT_BYTES = 4096;

  private static Server server;
  private static ManagedChannel channel;
  private static AnalysisServiceGrpc.AnalysisServiceStub async;
  private static AnalysisServiceGrpc.AnalysisServiceBlockingStub blocking;

  @BeforeAll
  static void setUp() throws Exception {
    final ServiceConfig config =
        new ServiceConfig(0, MAX_TEXT_BYTES, null, null, null, null, null, null);
    server = NettyServerBuilder.forPort(0)
        .addService(io.grpc.ServerInterceptors.intercept(
            new AnalysisServiceImpl(PipelineEnvironment.load(config), config),
            new EagerHeadersInterceptor()))
        .build()
        .start();
    channel = NettyChannelBuilder.forAddress("127.0.0.1", server.getPort())
        .usePlaintext()
        .build();
    async = AnalysisServiceGrpc.newStub(channel);
    blocking = AnalysisServiceGrpc.newBlockingStub(channel);
  }

  @AfterAll
  static void tearDown() throws Exception {
    channel.shutdownNow();
    server.shutdownNow();
    channel.awaitTermination(5, TimeUnit.SECONDS);
    server.awaitTermination(5, TimeUnit.SECONDS);
  }

  private static AnalysisOptions bm25Options() {
    return AnalysisOptions.newBuilder()
        .setTokenizer(AnalysisOptions.Tokenizer.TOKENIZER_SIMPLE)
        .setStemmer(AnalysisOptions.Stemmer.STEMMER_SNOWBALL_ENGLISH)
        .setTermVectors(TermVectorOptions.newBuilder()
            .setEnabled(true)
            .setMode(TermVectorOptions.Mode.MODE_FULL)
            .setSource(TermVectorOptions.Source.SOURCE_STEMS))
        .build();
  }

  private static AnalyzeStreamRequest options(AnalysisOptions options) {
    return AnalyzeStreamRequest.newBuilder().setOptions(options).build();
  }

  private static AnalyzeStreamRequest doc(long sequence, String text) {
    return AnalyzeStreamRequest.newBuilder()
        .setDoc(AnalyzeStreamDoc.newBuilder().setSequence(sequence).setText(text))
        .build();
  }

  /** Collects every response and the terminal signal of one stream. */
  private static final class Collector implements StreamObserver<AnalyzeStreamResponse> {
    final List<AnalyzeStreamResponse> responses = new CopyOnWriteArrayList<>();
    final AtomicReference<Throwable> error = new AtomicReference<>();
    final CountDownLatch done = new CountDownLatch(1);

    @Override
    public void onNext(AnalyzeStreamResponse response) {
      responses.add(response);
    }

    @Override
    public void onError(Throwable t) {
      error.set(t);
      done.countDown();
    }

    @Override
    public void onCompleted() {
      done.countDown();
    }

    void await() throws InterruptedException {
      assertThat(done.await(60, TimeUnit.SECONDS))
          .as("stream terminated within 60 s").isTrue();
    }
  }

  @Test
  void streamMatchesUnaryExactly() throws Exception {
    final AnalysisOptions options = bm25Options();
    final String[] texts = new String[40];
    for (int i = 0; i < texts.length; i++) {
      texts[i] = "The appellate court " + i + " affirmed the ruling; judges "
          + "were running proceedings " + "swiftly ".repeat(1 + i % 7) + "here.";
    }

    final Collector collector = new Collector();
    final StreamObserver<AnalyzeStreamRequest> requests = async.analyzeStream(collector);
    requests.onNext(options(options));
    for (int i = 0; i < texts.length; i++) {
      requests.onNext(doc(i, texts[i]));
    }
    requests.onCompleted();
    collector.await();

    assertThat(collector.error.get()).isNull();
    final Map<Long, AnalyzeStreamResponse> bySequence = collector.responses.stream()
        .collect(Collectors.toMap(AnalyzeStreamResponse::getSequence, r -> r));
    assertThat(bySequence).hasSize(texts.length);
    for (int i = 0; i < texts.length; i++) {
      final AnalyzeStreamResponse response = bySequence.get((long) i);
      assertThat(response.hasOk()).as("sequence %d is ok", i).isTrue();
      final AnalyzeResponse unary = blocking.analyze(AnalyzeRequest.newBuilder()
          .setText(texts[i]).setOptions(options).build());
      assertThat(response.getOk()).as("sequence %d equals unary", i).isEqualTo(unary);
    }
  }

  @Test
  void perDocumentErrorsKeepTheStreamAlive() throws Exception {
    final Collector collector = new Collector();
    final StreamObserver<AnalyzeStreamRequest> requests = async.analyzeStream(collector);
    requests.onNext(options(bm25Options()));
    requests.onNext(doc(1, "a fine document"));
    requests.onNext(doc(2, ""));
    requests.onNext(doc(3, "x".repeat(MAX_TEXT_BYTES + 1)));
    requests.onNext(doc(4, "still analyzed after the failures"));
    requests.onCompleted();
    collector.await();

    assertThat(collector.error.get()).isNull();
    final Map<Long, AnalyzeStreamResponse> bySequence = collector.responses.stream()
        .collect(Collectors.toMap(AnalyzeStreamResponse::getSequence, r -> r));
    assertThat(bySequence).hasSize(4);
    assertThat(bySequence.get(1L).hasOk()).isTrue();
    assertThat(bySequence.get(4L).hasOk()).isTrue();
    assertThat(bySequence.get(2L).hasError()).isTrue();
    assertThat(bySequence.get(2L).getError().getCode())
        .isEqualTo(Status.Code.INVALID_ARGUMENT.value());
    assertThat(bySequence.get(3L).hasError()).isTrue();
    assertThat(bySequence.get(3L).getError().getCode())
        .isEqualTo(Status.Code.INVALID_ARGUMENT.value());
    assertThat(bySequence.get(3L).getError().getMessage()).contains("above the limit");
  }

  @Test
  void firstMessageMustBeOptions() throws Exception {
    final Collector collector = new Collector();
    final StreamObserver<AnalyzeStreamRequest> requests = async.analyzeStream(collector);
    requests.onNext(doc(1, "no options came first"));
    collector.await();

    assertThat(collector.error.get()).isInstanceOf(StatusRuntimeException.class);
    assertThat(Status.fromThrowable(collector.error.get()).getCode())
        .isEqualTo(Status.Code.INVALID_ARGUMENT);
  }

  @Test
  void secondOptionsMessageFailsTheStream() throws Exception {
    final Collector collector = new Collector();
    final StreamObserver<AnalyzeStreamRequest> requests = async.analyzeStream(collector);
    requests.onNext(options(bm25Options()));
    requests.onNext(doc(1, "one document"));
    requests.onNext(options(bm25Options()));
    collector.await();

    assertThat(collector.error.get()).isInstanceOf(StatusRuntimeException.class);
    assertThat(Status.fromThrowable(collector.error.get()).getCode())
        .isEqualTo(Status.Code.INVALID_ARGUMENT);
    assertThat(Status.fromThrowable(collector.error.get()).getDescription())
        .contains("first message");
  }

  @Test
  void emptyMessageFailsTheStream() throws Exception {
    final Collector collector = new Collector();
    final StreamObserver<AnalyzeStreamRequest> requests = async.analyzeStream(collector);
    requests.onNext(options(bm25Options()));
    requests.onNext(AnalyzeStreamRequest.getDefaultInstance());
    collector.await();

    assertThat(collector.error.get()).isInstanceOf(StatusRuntimeException.class);
    assertThat(Status.fromThrowable(collector.error.get()).getCode())
        .isEqualTo(Status.Code.INVALID_ARGUMENT);
  }

  @Test
  void invalidOptionsFailTheStreamUpFront() throws Exception {
    // STEMS term identity without a stemmer is the option-set the mapper
    // rejects on the unary path; the stream must reject it identically.
    final AnalysisOptions invalid = AnalysisOptions.newBuilder()
        .setTermVectors(TermVectorOptions.newBuilder()
            .setEnabled(true)
            .setSource(TermVectorOptions.Source.SOURCE_STEMS))
        .build();
    final Collector collector = new Collector();
    final StreamObserver<AnalyzeStreamRequest> requests = async.analyzeStream(collector);
    requests.onNext(options(invalid));
    collector.await();

    assertThat(collector.error.get()).isInstanceOf(StatusRuntimeException.class);
    assertThat(Status.fromThrowable(collector.error.get()).getCode())
        .isEqualTo(Status.Code.INVALID_ARGUMENT);
  }

  @Test
  void manyDocumentsAllComplete() throws Exception {
    // Far more documents than the flow-control window: every grant path and
    // the completion race (client half-close vs in-flight workers) runs.
    final int docs = 500;
    final Collector collector = new Collector();
    final StreamObserver<AnalyzeStreamRequest> requests = async.analyzeStream(collector);
    requests.onNext(options(bm25Options()));
    for (int i = 0; i < docs; i++) {
      requests.onNext(doc(i, "document number " + i + " of the long stream"));
    }
    requests.onCompleted();
    collector.await();

    assertThat(collector.error.get()).isNull();
    assertThat(collector.responses).hasSize(docs);
    final var sequences = collector.responses.stream()
        .map(AnalyzeStreamResponse::getSequence).collect(Collectors.toSet());
    assertThat(sequences).hasSize(docs);
    assertThat(collector.responses).allMatch(AnalyzeStreamResponse::hasOk);
  }

  @Test
  void duplicateSequencesAreEchoedNotDeduplicated() throws Exception {
    final Collector collector = new Collector();
    final StreamObserver<AnalyzeStreamRequest> requests = async.analyzeStream(collector);
    requests.onNext(options(bm25Options()));
    requests.onNext(doc(7, "first use of the sequence"));
    requests.onNext(doc(7, "second use of the same sequence"));
    requests.onCompleted();
    collector.await();

    assertThat(collector.error.get()).isNull();
    assertThat(collector.responses).hasSize(2);
    assertThat(collector.responses)
        .allMatch(r -> r.getSequence() == 7L && r.hasOk());
  }

  @Test
  void headersArriveBeforeTheFirstResult() throws Exception {
    // A tonic client's stream open resolves on response HEADERS; grpc-java
    // sends them lazily with the first message unless the eager-headers
    // interceptor runs. This pins the interceptor: headers must arrive
    // while only options (no documents) have been sent, or an
    // open-before-submit client deadlocks.
    final CountDownLatch headers = new CountDownLatch(1);
    final io.grpc.ClientInterceptor recorder = new io.grpc.ClientInterceptor() {
      @Override
      public <ReqT, RespT> io.grpc.ClientCall<ReqT, RespT> interceptCall(
          io.grpc.MethodDescriptor<ReqT, RespT> method, io.grpc.CallOptions options,
          io.grpc.Channel next) {
        return new io.grpc.ForwardingClientCall.SimpleForwardingClientCall<>(
            next.newCall(method, options)) {
          @Override
          public void start(Listener<RespT> listener, io.grpc.Metadata metadata) {
            super.start(
                new io.grpc.ForwardingClientCallListener
                    .SimpleForwardingClientCallListener<>(listener) {
                  @Override
                  public void onHeaders(io.grpc.Metadata received) {
                    headers.countDown();
                    super.onHeaders(received);
                  }
                }, metadata);
          }
        };
      }
    };
    final Collector collector = new Collector();
    final StreamObserver<AnalyzeStreamRequest> requests =
        AnalysisServiceGrpc.newStub(channel).withInterceptors(recorder)
            .analyzeStream(collector);
    requests.onNext(options(bm25Options()));
    assertThat(headers.await(10, TimeUnit.SECONDS))
        .as("headers before any document was submitted").isTrue();
    requests.onCompleted();
    collector.await();
    assertThat(collector.error.get()).isNull();
  }

  @Test
  void concurrentStreamsShareThePipelineCache() throws Exception {
    final int streams = 4;
    final ConcurrentHashMap<Integer, Collector> collectors = new ConcurrentHashMap<>();
    for (int s = 0; s < streams; s++) {
      final Collector collector = new Collector();
      collectors.put(s, collector);
      final StreamObserver<AnalyzeStreamRequest> requests = async.analyzeStream(collector);
      requests.onNext(options(bm25Options()));
      for (int i = 0; i < 25; i++) {
        requests.onNext(doc(i, "stream " + s + " document " + i));
      }
      requests.onCompleted();
    }
    for (final Collector collector : collectors.values()) {
      collector.await();
      assertThat(collector.error.get()).isNull();
      assertThat(collector.responses).hasSize(25);
      assertThat(collector.responses).allMatch(AnalyzeStreamResponse::hasOk);
    }
  }
}

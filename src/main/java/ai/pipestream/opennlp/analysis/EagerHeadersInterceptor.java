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

import io.grpc.ForwardingServerCall;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;

/**
 * Sends response headers as soon as a call is accepted instead of lazily
 * with the first response message (grpc-java's default).
 *
 * <p>This is load-bearing for AnalyzeStream: a client that awaits call
 * acceptance before submitting documents (tonic's stream open resolves on
 * response headers) would otherwise deadlock against a server whose first
 * response message only exists once a document has been submitted. Eager
 * headers break the cycle: acceptance is visible before any result.</p>
 */
final class EagerHeadersInterceptor implements ServerInterceptor {

  @Override
  public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
      ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
    final ForwardingServerCall.SimpleForwardingServerCall<ReqT, RespT> wrapped =
        new ForwardingServerCall.SimpleForwardingServerCall<>(call) {
          private boolean sent;

          @Override
          public void sendHeaders(Metadata responseHeaders) {
            if (!sent) {
              sent = true;
              super.sendHeaders(responseHeaders);
            }
          }
        };
    final ServerCall.Listener<ReqT> listener = next.startCall(wrapped, headers);
    // After startCall so the handler is fully wired; the wrapper turns the
    // stub's own (now second) sendHeaders into a no-op.
    wrapped.sendHeaders(new Metadata());
    return listener;
  }
}

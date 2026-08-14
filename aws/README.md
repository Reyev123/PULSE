# PULSE AWS deployment

## Architecture

```text
Browser -> public ALB (HTTP :80) -> ECS Fargate service (1 task) -> Dash GUI
                                             |
                                             -> ephemeral container memory/tmp
                                             |
                                             -> optional private Ollama EC2 host
```

This is intentionally small:

- ECS Fargate, normally one task (`DesiredCount=1`)
- Application Load Balancer in two public subnets
- ECR image repository with scan-on-push and five-image retention
- CloudWatch logs retained for 14 days
- No database, S3 bucket, user IDs, authentication, or persistent uploads
- Training data is not shipped to AWS; it is not needed at runtime and is about 1.4 GB
- Optional dedicated Ollama EC2 host pulls `llama3.1:8b` (~5 GB) at boot

The service is public over HTTP and has no authentication by request. This is
appropriate only for a controlled, limited-user demonstration. Do not use it
for protected health information or clinical production without adding TLS,
access control, audit logging, and a security review.

## Runtime behavior

The image contains the Dash GUI, ECG analysis code, ECG digitizer, RhythmCNN
checkpoint, and rendering helper. Uploads are processed in the task and are
not persisted. A task replacement loses in-memory state, as expected.

The optional Ollama host is a separate CPU-only `t3.xlarge` by default. It is
more reliable than putting a 5 GB model inside the web task, but generation can
be slow. A GPU EC2 instance can be selected later after checking regional GPU
capacity and quota. The model is downloaded once onto that instance's local
disk; it is not copied into the web image.

Without Ollama, the GUI displays its unavailable-model message. Signal
analysis, plots, RhythmCNN, and PDF generation do not require Ollama.
Ollama is optional at runtime: if its EC2 instance is stopped, the application
does not start it and safely returns deterministic measurements without the
narrative.

## Prerequisites

1. Revoke/rotate any AWS keys that were pasted into chat or shell history.
2. Install Docker and AWS CLI.
3. Configure a named AWS CLI profile using an IAM role or short-lived credentials:

```bash
aws configure --profile pulse-deploy
aws sts get-caller-identity --profile pulse-deploy
```

Do not commit credentials, `.env` files, access keys, or session tokens.

The deployment settings are in `aws/config.env.example`. Copy it to
`aws/config.env`, edit the profile/region/model settings, and keep the local
file uncommitted. `deploy.sh` also accepts the equivalent `PULSE_*` environment
variables.

## Deploy

For the complete lowest-cost setup, follow
[DEPLOY_FULL_SYSTEM.md](DEPLOY_FULL_SYSTEM.md). It includes ACM and IONOS DNS
steps and enables the Ollama host.

Run from the repository root, using a profile that can manage CloudFormation,
ECR, ECS, ELB, EC2 networking, IAM task roles, and CloudWatch Logs:

```bash
source aws/config.env
./aws/deploy.sh
```

The script first creates the network/ECR/ALB with zero tasks, then builds and
pushes the image, and finally starts the Fargate service. It prints the public
ALB URL at the end. The first deployment can take several minutes.

Useful overrides:

```bash
PULSE_STACK_NAME=pulse-ecg-demo PULSE_DESIRED_COUNT=1 AWS_PROFILE=pulse-deploy AWS_REGION=us-east-1 ./aws/deploy.sh
```

To run narrative generation on AWS, set `PULSE_ENABLE_OLLAMA=true`. The
CloudFormation stack launches the model host and injects its private address
into the ECS task as `OLLAMA_URL`. This adds EC2 charges and CPU inference
latency. Set it to `false` when only deterministic ECG analysis is needed.

### Cheaper/faster LLM: fall back to a local Spark Ollama

The CPU Ollama VM is slow. To stay on budget you can keep it stopped (or off
entirely) and let the deployment fall back to a faster local machine's Ollama
(e.g. the Spark box) when it is reachable:

1. On the Spark machine, expose Ollama on all interfaces and pull the model:
   `OLLAMA_HOST=0.0.0.0:11434 ollama serve` and `ollama pull llama3.1:8b`.
2. Make it reachable from AWS over a public URL or tunnel (e.g. an SSH reverse
   tunnel, Tailscale, or a restricted public port). Note the resulting URL,
   e.g. `http://<spark-host>:11434`.
3. Set the fallback and deploy:

   ```bash
   PULSE_OLLAMA_FALLBACK_URL=http://<spark-host>:11434 ./aws/deploy.sh
   ```

The app tries the AWS VM first (`OLLAMA_URL`), then each `OLLAMA_FALLBACK_URL`
in order, using the first that serves the model. If the AWS VM is stopped, the
Spark endpoint is used automatically. If none respond, the report still returns
the deterministic measurements. Only expose the Spark endpoint to trusted
networks; it is unauthenticated.

### Recommended budget path: Tailscale sidecar to the Spark Ollama

Instead of exposing Ollama publicly, add a **Tailscale sidecar** to the ECS
task so it reaches the Spark Ollama privately over your tailnet. The app talks
to Ollama through the sidecar's SOCKS5 proxy (`OLLAMA_SOCKS5=127.0.0.1:1055`,
set automatically when the sidecar is on). With this you can turn the EC2 model
host off entirely (`PULSE_ENABLE_OLLAMA=false`) and keep costs near zero.

Manage the Spark Ollama with the helper scripts:

```bash
./aws/ollama-local-start.sh   # serve on 0.0.0.0:11434, pull model, print tailnet URL
./aws/ollama-local-stop.sh    # stop it
```

One-time setup, then deploy:

1. In the Tailscale admin console, create an **ephemeral, reusable** auth key.
2. Store it as an SSM SecureString (type the key directly; never paste it into
   chat or commit it):

   ```bash
   aws ssm put-parameter --name /pulse/tailscale-authkey --type SecureString \
     --value 'tskey-...' --profile pulse-deploy --region us-east-1
   ```

3. Note the Spark tailnet IP (`tailscale ip -4`, e.g. `100.84.111.7`) and deploy:

   ```bash
   PULSE_ENABLE_TAILSCALE=true PULSE_ENABLE_OLLAMA=false \
   PULSE_OLLAMA_URL=http://<spark-tailnet-ip>:11434 \
   ./aws/deploy.sh
   ```

The sidecar container is `Essential: false`: if it or the Spark Ollama is down,
the app stays up and returns deterministic measurements without the narrative.
Rotate/revoke the auth key in the Tailscale admin console when no longer needed.

## `pulse.superraev.com` with IONOS

1. Request an ACM certificate in the same AWS region as the stack for
  `pulse.superraev.com`, using DNS validation.
2. In IONOS DNS management, create the ACM validation CNAME record exactly as
  AWS provides it. Wait until ACM shows the certificate as **Issued**.
3. Put the issued certificate ARN in `PULSE_CERTIFICATE_ARN` in
  `aws/config.env` and deploy.
4. The stack output includes the ALB DNS name. In IONOS create:
  - Host/name: `pulse`
  - Type: `CNAME`
  - Target: the ALB DNS name, for example
    `pulse-ecg-123456.us-east-1.elb.amazonaws.com`
5. Browse to `https://pulse.superraev.com` after DNS propagation.

The supplied IONOS login URL is only the account login page; DNS changes must
be made inside the authenticated IONOS domain/DNS management console. Never
send IONOS credentials or AWS keys through this chat.

## Operations

View recent application logs:

```bash
aws logs tail /ecs/pulse-ecg --follow --profile pulse-deploy --region us-east-1
```

Scale to zero when not in use:

```bash
aws ecs update-service --cluster pulse-ecg --service pulse-ecg --desired-count 0 --profile pulse-deploy --region us-east-1
```

Delete the stack when finished. The ECR repository and log group are retained
by design, so remove those separately if they are no longer needed.

## Production follow-ups

- Add HTTPS with ACM and an HTTPS ALB listener.
- Restrict the ALB security group to known IP ranges, or add authentication.
- Put Ollama behind a separately secured service, or replace it with a managed
  model endpoint such as Amazon Bedrock.
- Add CloudWatch alarms and a WAF if the endpoint becomes internet-facing.
- Keep raw ECG uploads out of logs and persistent storage.

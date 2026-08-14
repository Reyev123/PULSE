#!/usr/bin/env bash
set -euo pipefail

# Use a configured AWS profile or an IAM role; never put keys in this script.
PROFILE="${AWS_PROFILE:-default}"
REGION="${AWS_REGION:-us-east-1}"
STACK="${PULSE_STACK_NAME:-pulse-ecg}"
OLLAMA_URL="${PULSE_OLLAMA_URL:-http://127.0.0.1:11434}"
OLLAMA_FALLBACK_URL="${PULSE_OLLAMA_FALLBACK_URL:-}"
OLLAMA_MODEL="${PULSE_OLLAMA_MODEL:-llama3.1:8b}"
ENABLE_OLLAMA="${PULSE_ENABLE_OLLAMA:-false}"
ENABLE_TAILSCALE="${PULSE_ENABLE_TAILSCALE:-false}"
TAILSCALE_AUTHKEY_PARAM="${PULSE_TAILSCALE_AUTHKEY_PARAM:-/pulse/tailscale-authkey}"
OLLAMA_INSTANCE_TYPE="${PULSE_OLLAMA_INSTANCE_TYPE:-t3a.xlarge}"
CERTIFICATE_ARN="${PULSE_CERTIFICATE_ARN:-}"
DOMAIN_NAME="${PULSE_DOMAIN_NAME:-pulse.superraev.com}"
TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cloudformation.yml"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLACEHOLDER="public.ecr.aws/docker/library/python:3.10-slim"

if [[ -f "$(dirname "${BASH_SOURCE[0]}")/config.env" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "${BASH_SOURCE[0]}")/config.env"
  PROFILE="${AWS_PROFILE:-$PROFILE}"
  REGION="${AWS_REGION:-$REGION}"
  STACK="${PULSE_STACK_NAME:-$STACK}"
  OLLAMA_URL="${PULSE_OLLAMA_URL:-$OLLAMA_URL}"
  OLLAMA_FALLBACK_URL="${PULSE_OLLAMA_FALLBACK_URL:-$OLLAMA_FALLBACK_URL}"
  OLLAMA_MODEL="${PULSE_OLLAMA_MODEL:-$OLLAMA_MODEL}"
  ENABLE_OLLAMA="${PULSE_ENABLE_OLLAMA:-$ENABLE_OLLAMA}"
  ENABLE_TAILSCALE="${PULSE_ENABLE_TAILSCALE:-$ENABLE_TAILSCALE}"
  TAILSCALE_AUTHKEY_PARAM="${PULSE_TAILSCALE_AUTHKEY_PARAM:-$TAILSCALE_AUTHKEY_PARAM}"
  OLLAMA_INSTANCE_TYPE="${PULSE_OLLAMA_INSTANCE_TYPE:-$OLLAMA_INSTANCE_TYPE}"
  CERTIFICATE_ARN="${PULSE_CERTIFICATE_ARN:-$CERTIFICATE_ARN}"
  DOMAIN_NAME="${PULSE_DOMAIN_NAME:-$DOMAIN_NAME}"
fi

aws_cmd=(aws --profile "$PROFILE" --region "$REGION")

echo "Checking AWS identity for profile: $PROFILE"
"${aws_cmd[@]}" sts get-caller-identity

echo "Creating infrastructure with zero tasks so the ECR repository exists..."
"${aws_cmd[@]}" cloudformation deploy \
  --stack-name "$STACK" \
  --template-file "$TEMPLATE" \
  --parameter-overrides ImageUri="$PLACEHOLDER" DesiredCount=0 \
    OllamaUrl="$OLLAMA_URL" OllamaModel="$OLLAMA_MODEL" \
    OllamaFallbackUrl="$OLLAMA_FALLBACK_URL" \
    EnableOllama="$ENABLE_OLLAMA" OllamaInstanceType="$OLLAMA_INSTANCE_TYPE" \
    EnableTailscale="$ENABLE_TAILSCALE" TailscaleAuthKeyParam="$TAILSCALE_AUTHKEY_PARAM" \
    CertificateArn="$CERTIFICATE_ARN" DomainName="$DOMAIN_NAME" \
  --capabilities CAPABILITY_NAMED_IAM

REPO_URI=$("${aws_cmd[@]}" cloudformation describe-stacks \
  --stack-name "$STACK" \
  --query 'Stacks[0].Outputs[?OutputKey==`EcrRepositoryUri`].OutputValue' \
  --output text)
IMAGE_URI="$REPO_URI:${PULSE_IMAGE_TAG:-$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"

echo "Building runtime image: $IMAGE_URI"
docker build --platform linux/arm64 --file "$ROOT/aws/Dockerfile" --tag "$IMAGE_URI" "$ROOT"

"${aws_cmd[@]}" ecr get-login-password | docker login --username AWS --password-stdin "$REPO_URI"
docker push "$IMAGE_URI"

echo "Starting ECS service..."
"${aws_cmd[@]}" cloudformation deploy \
  --stack-name "$STACK" \
  --template-file "$TEMPLATE" \
  --parameter-overrides ImageUri="$IMAGE_URI" DesiredCount="${PULSE_DESIRED_COUNT:-1}" \
    OllamaUrl="$OLLAMA_URL" OllamaModel="$OLLAMA_MODEL" \
    OllamaFallbackUrl="$OLLAMA_FALLBACK_URL" \
    EnableOllama="$ENABLE_OLLAMA" OllamaInstanceType="$OLLAMA_INSTANCE_TYPE" \
    EnableTailscale="$ENABLE_TAILSCALE" TailscaleAuthKeyParam="$TAILSCALE_AUTHKEY_PARAM" \
    CertificateArn="$CERTIFICATE_ARN" DomainName="$DOMAIN_NAME" \
  --capabilities CAPABILITY_NAMED_IAM

echo "Application URL:"
"${aws_cmd[@]}" cloudformation describe-stacks \
  --stack-name "$STACK" \
  --query 'Stacks[0].Outputs[?OutputKey==`ApplicationUrl`].OutputValue' \
  --output text

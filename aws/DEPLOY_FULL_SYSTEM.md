# Deploy the lowest-cost full PULSE system

This runbook deploys the complete system:

- one ECS Fargate web task: 1 vCPU / 2 GB RAM;
- one `t3a.xlarge` EC2 Ollama host: 4 vCPU / 16 GB RAM;
- `llama3.1:8b` downloaded automatically on the Ollama host;
- an internet-facing Application Load Balancer;
- HTTPS for `pulse.superraev.com`;
- no database, authentication, user IDs, or persistent ECG uploads.

Expected baseline cost is approximately **$165–190/month** in `us-east-1` when both compute services remain running. This is an estimate; check the AWS billing console after deployment.

## 1. Credentials

Configure the profile locally, without committing credentials:

```bash
aws configure --profile pulse-deploy
aws sts get-caller-identity --profile pulse-deploy
```

The identity command must return the intended AWS account before continuing.

## 2. Create the ACM certificate

The certificate must be created in the **same AWS region** used for the stack. For the example configuration, use `us-east-1`.

1. Open AWS Certificate Manager (ACM).
2. Choose **Request a public certificate**.
3. Add the name `pulse.superraev.com`.
4. Choose **DNS validation**.
5. Submit the request.
6. Open the certificate's validation details and copy the generated CNAME:
    - CNAME name, usually beginning with `_`;
    - CNAME value, usually ending in `acm-validations.aws.`.

Do not guess these values. They are unique to the certificate request.

## 3. Add ACM validation DNS records at IONOS

In the IONOS domain/DNS management console for `superraev.com`, add the ACM validation record exactly as AWS displays it:

| Setting | Value |
|---|---|
| Host/name | The ACM-generated CNAME name, without the superraev.com suffix if IONOS adds it automatically |
| Type | CNAME |
| Target/value | The ACM-generated CNAME value |
| TTL | Default or 300 seconds |

Avoid creating an extra trailing `superraev.com` if the IONOS editor appends the zone automatically. Wait until ACM reports **Issued**. This can take several minutes.

## 4. Prepare the deployment settings

Copy the example file to the ignored local configuration filename:

```bash
cp aws/full-system.env.example aws/config.env
```

Edit `aws/config.env` and set:

- `AWS_PROFILE` to the profile created in step 1;
- `AWS_REGION` to the ACM certificate's region;
- `PULSE_CERTIFICATE_ARN` to the issued ACM certificate ARN;
- `PULSE_DOMAIN_NAME` to `pulse.superraev.com`.

Keep these values:

- `PULSE_DESIRED_COUNT=1`;
- `PULSE_ENABLE_OLLAMA=true`;
- `PULSE_OLLAMA_INSTANCE_TYPE=t3a.xlarge`;
- `PULSE_OLLAMA_MODEL=llama3.1:8b`.

Optional (budget): to fall back to a faster local Spark Ollama when the AWS VM
is stopped, also set `PULSE_OLLAMA_FALLBACK_URL` to the Spark machine's
reachable Ollama URL, e.g. `http://<spark-host>:11434`. You may then set
`PULSE_ENABLE_OLLAMA=false` to skip the EC2 model host entirely and rely on the
Spark endpoint. See "Cheaper/faster LLM" in [README.md](README.md).

Never add access keys, secret keys, or session tokens to `aws/config.env`. The AWS CLI profile supplies credentials separately.

## 5. Deploy

From the repository root:

```bash
source aws/config.env
./aws/deploy.sh
```

The script will:

1. verify the AWS identity;
2. create the VPC, ALB, ECR repository, ECS cluster, Ollama EC2 host, and logs;
3. build and push the runtime image;
4. start one ECS task;
5. configure ECS to use the private Ollama host;
6. print the ALB URL.

The image is built explicitly for `linux/arm64`, matching the ARM64 ECS
Fargate runtime and allowing native builds on this ARM64 workstation. This is
also the Graviton-compatible, lower-cost Fargate option.

The first deployment may take 10–20 minutes. Ollama then downloads the model; wait for the EC2 user-data process to finish before testing **Run Inference**. The application works for ECG analysis before narrative generation is ready.

### If the Ollama EC2 instance is stopped

The application does **not** start, restart, or control the Ollama instance. If the instance is stopped or the model is still loading:

- file upload and ECG signal analysis continue;
- RhythmCNN classification continues;
- graphs and trends continue;
- PDF export continues;
- **Run Inference** returns the deterministic measurements with an explicit “Narrative unavailable” notice;
- the request fails fast after the configured three-second Ollama check rather than hanging or causing ECS to restart the instance.

To restore narrative generation, start the EC2 instance manually and wait for the Ollama model pull/readiness process. No ECS redeploy is required while the private address remains unchanged.

Check deployment status:

```bash
aws cloudformation describe-stacks --stack-name "$PULSE_STACK_NAME" --profile "$AWS_PROFILE" --region "$AWS_REGION"
aws ecs list-tasks --cluster pulse-ecg --profile "$AWS_PROFILE" --region "$AWS_REGION"
aws logs tail /ecs/pulse-ecg --follow --profile "$AWS_PROFILE" --region "$AWS_REGION"
```

## 6. Add the application DNS record at IONOS

After deployment, obtain the ALB DNS name from CloudFormation outputs:

```bash
aws cloudformation describe-stacks
  --stack-name "$PULSE_STACK_NAME"
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDnsName`].OutputValue'
  --output text
  --profile "$AWS_PROFILE"
  --region "$AWS_REGION"
```

In IONOS DNS management for `superraev.com`, add:

| Setting | Value |
|---|---|
| Host/name | pulse |
| Type | CNAME |
| Target/value | The ALB DNS name, for example pulse-ecg-123456.us-east-1.elb.amazonaws.com |
| TTL | Default or 300 seconds |

Do not use an A record with a fixed ALB IP. ALB IP addresses can change.

After DNS propagation, open:

`https://pulse.superraev.com`

HTTP requests redirect to HTTPS when `PULSE_CERTIFICATE_ARN` is configured.

## 7. Verify the full system

1. Open `https://pulse.superraev.com`.
2. Upload a CSV sample.
3. Confirm the ECG graphs and RhythmCNN result appear.
4. Click **Run Inference**.
5. Confirm the narrative no longer reports that Ollama is unavailable.
6. Export a PDF.

If the narrative is unavailable, inspect the Ollama host system log and wait for the model pull to complete. The Ollama host is intentionally restricted so only the ECS security group can call port `11434`.

## Cost controls

To stop application traffic while retaining the stack, scale ECS to zero:

```bash
aws ecs update-service --cluster pulse-ecg --service pulse-ecg --desired-count 0 --profile "$AWS_PROFILE" --region "$AWS_REGION"
```

This does **not** stop the Ollama EC2 instance. Stop the instance separately when it is not needed; restarting it may require another model readiness wait. The ALB still incurs a charge while retained.

To remove everything managed by the stack:

```bash
aws cloudformation delete-stack --stack-name "$PULSE_STACK_NAME" --profile "$AWS_PROFILE" --region "$AWS_REGION"
```

The ECR repository and CloudWatch log group are retained by design and must be removed separately if no longer needed.

## Security and limitations

This is a limited-user research/demo deployment with no authentication. Do not upload protected health information. Add authentication, access restrictions, audit logging, WAF, and operational monitoring before clinical or public use.
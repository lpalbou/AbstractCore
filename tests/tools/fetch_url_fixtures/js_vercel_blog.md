# Vercel Blog

> The Vercel blog covers engineering, product updates, and customer stories.
> Each post is available as markdown at /blog/{slug}

Total posts: 593

---

### [How v0 authenticates to Snowflake without exposing the user's OAuth token](/blog/how-v0-authenticates-to-snowflake-without-exposing-the-users-oauth-token)
Published: 2026-08-20
Generated code in v0 sandboxes queries Snowflake, but the OAuth token never enters the sandbox. A protocol-aware proxy injects credentials at request time, and blind replacement was itself a leak.

### [Introducing Vercel for Slack](/blog/introducing-vercel-for-slack)
Published: 2026-08-19
Vercel Agent now works in Slack. Mention @Vercel to diagnose incidents, review PRs, and ship approved changes without leaving the conversation. In public beta for Pro and Enterprise teams.

### [$1 million hacker challenge for Vercel Sandbox](/blog/one-million-dollar-hacker-challenge-for-vercel-sandbox)
Published: 2026-08-18
Category: Security
Vercel is running a two-week, public HackerOne program with up to $1,000,000 in bounties for researchers who can escape a Vercel Sandbox.

### [Inside the Vercel intern experience](/blog/inside-the-vercel-intern-experience)
Published: 2026-08-13
Meet Vercel’s winter 2026 interns and see how they shipped production features across the CDN, v0, financial infrastructure, AI Gateway, and more.

### [Building a software factory for AI SDK](/blog/building-a-software-factory-for-ai-sdk)
Published: 2026-08-12
Category: Engineering
We built a software factory that autonomously processes issues and PRs for the AI SDK, with humans in control of every merge. Four weeks in, it authors 25-40% of merged PRs.

### [How we migrated the database behind every Vercel build](/blog/how-we-migrated-the-database-behind-every-vercel-build)
Published: 2026-08-11
How we migrated the build warm pool, one of the most critical services at Vercel, from Redis to DynamoDB live in production, through feature-flagged phases with a rollback at every step.

### [Everything hackable will get hacked](/blog/everything-hackable-will-get-hacked)
Published: 2026-08-11
Open-weight models that allow perform offensive research present an emerging cybersecurity challenge. But you do not need to wait for specialized models to begin defensive cybersecurity work.

### [DeepSeek overtakes Google on volume, cost per token falls 13.6%](/blog/deepseek-overtakes-google-on-volume-cost-per-token-falls)
Published: 2026-08-11
The AI Gateway production index covering data from July 2026: DeepSeek became the second-largest lab by token volume; Anthropic held 65% of spend at more than twice the average price per token

### [A sandbox without a network boundary is only half a sandbox](/blog/a-sandbox-without-a-network-boundary-is-only-half-a-sandbox)
Published: 2026-08-11
Category: Engineering
Running untrusted code safely requires more than separating it from the host. You also have to control what that code can reach.

### [Introducing Agent Plugins](/blog/introducing-agent-plugins)
Published: 2026-08-06
Agent Plugins 1.0.0 is an open, vendor-neutral specification for packaging Agent Skills and MCP servers into distributable plugins that compatible AI agent clients can discover and load.

### [Introducing the new v0 API](/blog/introducing-the-new-v0-api)
Published: 2026-08-05
Category: Field Engineering
Today we're introducing the new v0 API: programmatic, headless access to v0's app-building agent. Send a prompt and v0 generates an app, starts a dev server in a Vercel Sandbox, and gives you a preview URL you can embed in your own UI.

### [Next.js 16.3 support on Vercel](/blog/vercel-supports-next-js-16-3)
Published: 2026-08-04
Category: Engineering
Next.js 16.3 applications on Vercel send 45% fewer prefetch requests, 17% fewer static assets, and have 2x faster path metadata serving.

### [How Factory scaled its cloud backend to one billion monthly requests on Vercel](/customers/how-factory-scaled-its-cloud-backend-to-one-billion-monthly-requests-on-vercel)
Published: 2026-08-03
Category: Customer stories
Factory runs the largest api.* cloud backend on Vercel, serving one billion requests monthly. See how they scaled three years of backend workloads, deployed internal tools autonomously, and stopped fraud without building a security team.

### [Shopify and Vercel are rebuilding Hydrogen for faster storefronts](/customers/shopify-and-vercel-are-rebuilding-hydrogen-for-faster-storefronts)
Published: 2026-07-30
Category: Customer stories
Shopify powers millions of merchants. Rebuilding Hydrogen with Vercel brings agentic commerce to every storefront, and cuts feature work from months to a week.

### [How Sandstone grew 40x in 147 days on Vercel](/customers/how-sandstone-grew-40x-in-147-days-on-vercel)
Published: 2026-07-27
Category: Customer stories
Sandstone is the Legal Relationship Management system that coordinates a delightful day-to-day for legal departments.

### [DeepsecBench: evaluating model performance in finding cybersecurity vulnerabilities](/blog/deepsecbench-evaluating-model-performance-in-finding-cybersecurity-vulnerabilities)
Published: 2026-07-27
Category: Security
Today we're releasing DeepsecBench, a benchmark that evaluates how well different models find security vulnerabilities in application code. 

### [How Searchable ships customer-requested features in 30 minutes on Vercel ](/customers/how-searchable-ships-customer-requested-features-in-30-minutes-on-vercel)
Published: 2026-07-21
Category: Customer stories
Searchable tracks how brands show up in ChatGPT, Perplexity, and Google. On Vercel's AI SDK and AI Gateway, the lean team ships new features in 30 minutes.

### [Introducing the new Vercel Agent](/blog/vercel-agent)
Published: 2026-07-21
Category: Company News
Today we are expanding access to Vercel Agent: an AI agent that safely and autonomously investigates incidents, fixes builds, and reviews PRs on your production

### [How Speechify serves 500,000 dynamic pages to 60 million users on Vercel](/customers/how-speechify-serves-50000-dynamic-pages-to-60-million-users-on-vercel)
Published: 2026-07-15
Category: Customer stories
Speechify serves 500,000 dynamic pages across 45 languages to 60 million users on Vercel. With Cache Components, ISR, and Instant Rollbacks, their growth team ships twice a day without breaking anything.

### [Open-weight models surge to 29% of volume, price per token flattens](/blog/ai-gateway-production-index-july-2026)
Published: 2026-07-13
Category: Company News
The AI Gateway production index covering data from June 2026: Open-weight models increased to a third of all token volume; Anthropic continues to dominate spend with 61%

### [Vercel acquires Better Auth to accelerate open source auth](/blog/vercel-acquires-better-auth)
Published: 2026-07-07
Category: Company News
Vercel is acquiring Better Auth, the open source TypeScript auth library. It stays free and MIT licensed as the team builds identity for the agent era.

### [Run any Dockerfile on Vercel](/blog/dockerfile-on-vercel)
Published: 2026-06-30
Vercel now runs any HTTP server straight from a Dockerfile. Bring a Rails, Django, Spring Boot, or Go app and Vercel builds, deploys, autoscales, and runs it on Fluid compute with Active CPU pricing.

### [Vercel Ship 2026 recap](/blog/vercel-ship-2026-recap)
Published: 2026-06-30
Category: Company News
For a decade, Vercel has shaped how the web gets built. Now, we’re doing the same for agents. Vercel Ship 2026 brought over 2,500 people to London to build on Agentic Infrastructure.

### [Vercel and Shopify are rebuilding Hydrogen](/blog/vercel-and-shopify-are-rebuilding-hydrogen)
Published: 2026-06-30
Vercel commits to being a design partner on the open source rebuild of Hydrogen, Shopify's framework for headless storefronts.

### [Vercel Services: Run full stack on Vercel](/blog/vercel-services-run-full-stack-on-vercel)
Published: 2026-06-30
Run your whole backend on Vercel. Vercel Services deploys your frontend and every backend as one project, with routing, environment variables, and internal service-to-service communication wired automatically.

### [Vercel Open Source Program: Spring 2026 cohort](/blog/vercel-open-source-program-spring-2026-cohort)
Published: 2026-06-29
Category: Community
Announcing the spring 2026 cohort of Vercel's Open Source Program. Open source community tools, frameworks, and libraries

### [Build realtime voice agents on AI Gateway](/blog/realtime-voice-agents-on-ai-gateway)
Published: 2026-06-29
Audio/voice support is live on AI Gateway, with realtime. Build realtime, low-latency voice agents with the AI SDK on Vercel AI Gateway, plus text to speech and speech to text behind one API key and your existing tooling.

### [AI SDK 7](/blog/ai-sdk-7)
Published: 2026-06-25
AI SDK is the TypeScript SDK for building AI applications, features, frameworks, and agents across any model provider. AI SDK 7 focuses on what it takes to run AI in production.

### [Teaching agents product design at Vercel](/blog/teaching-agents-product-design-at-vercel)
Published: 2026-06-25
Learn how Vercel teaches agents product design with agent skills, lint rules, Vercel Agent code reviews, evals, and a human-led update loop.

### [Vercel Flags: Platform-native feature flags](/blog/vercel-flags-platform-native-feature-flags)
Published: 2026-06-22
Vercel Flags is a platform-native feature flag provider built into the Vercel developer platform, server-side by default with zero impact on page performance.

### [The Agent Stack](/blog/agent-stack)
Published: 2026-06-17
Category: Company News
Build production-grade AI agents with Vercel's Agent Stack. Connect to any model, run durable multi-step workflows, and securely link agents to your data and tools. Every building block you need, in one place.

### [Introducing Vercel Connect](/blog/introducing-vercel-connect)
Published: 2026-06-17
Vercel Connect lets your apps and agents access Slack, GitHub, and other services without storing long-lived secrets. Register a connector once and request scoped, short-lived tokens at runtime. Now in Public Beta.

### [Introducing eve](/blog/introducing-eve)
Published: 2026-06-17
Category: Company News
Introducing eve, the open-source agent framework from Vercel for building, running, and scaling agents in production, with durable execution, sandboxed compute, approvals, channels, tracing, and evals built in.

### [Vercel for Enterprise Apps and Agents](/blog/vercel-for-enterprise-apps-and-agents)
Published: 2026-06-16
Category: Company News
Vercel for Enterprise Apps and Agents gives your entire company the ability to ship apps and agents safely, with identity, credential scoping, and infrastructure controls built into the platform by default.

### [How Okara runs CMO agents for 120,000 companies on Vercel](/customers/how-okara-runs-cmo-agents-for-120000-companies-on-vercel)
Published: 2026-06-11
Category: Customer stories
Okara built an AI CMO on Vercel that directs eight specialized agents to handle SEO, content, and social for 120,000+ businesses, powered by a team of four using Vercel AI Gateway and Sandboxes.

### [How the Weather Company serves real-time forecasts to 350 million daily active users on Vercel](/customers/how-the-weather-company-serves-real-time-forecasts-to-350-million-daily-active-users-on-vercel)
Published: 2026-06-09
Category: Customer stories
The Weather Company rebuilt its web stack and CMS on Vercel and v0, cutting design-to-launch from days to hours and lifting productivity 80%.

### [How Code and Theory cut time-to-prototype 75% with v0](/customers/how-code-and-theory-cut-time-to-prototype-75-with-v0)
Published: 2026-06-09
Category: Customer stories
Code and Theory cut time-to-prototype by 75% and deployment timelines 50 to 75% by replacing wireframes and requirement docs with prompt-to-code in v0.

### [How Fern runs multi-tenant docs for Webflow and ElevenLabs on Vercel ](/customers/how-fern-runs-multi-tenant-docs-for-webflow-and-elevenlabs-on-vercel)
Published: 2026-06-09
Category: Customer stories
Fern runs multi-tenant docs for Webflow and ElevenLabs on Vercel, serving 6M+ monthly page views with 50 to 80% faster page loads and 3x faster TTFB.

### [DeepSeek enters the fight for token volume, Anthropic continues to dominate spend](/blog/ai-gateway-production-index-june-2026)
Published: 2026-06-08
Category: Company News
The June 2026 AI Gateway production index: DeepSeek's token share jumped to 17% as low-cost models entered production, while Anthropic held 65% of all spend.

### [Protecting against token theft](/blog/protecting-against-token-theft)
Published: 2026-05-29
Category: Field Engineering
Inference theft lets attackers resell your paid AI calls. See how the attack works, why rate limits and auth walls fail, and how Vercel BotID stops it on every request.

### [How Conductor moved parallel coding agents from the laptop to the cloud with Vercel Sandbox](/customers/how-conductor-moved-parallel-coding-agents-from-the-laptop-to-the-cloud-with-vercel-sandbox)
Published: 2026-05-27
Category: Customer stories
Conductor built Cloud Workspaces on Vercel Sandbox so    developers can run a fleet of coding agents in parallel,    close the laptop, and have the agents keep working. Used by    teams at Notion, Linear, and Ramp.

### [AI Gateway production index](/blog/ai-gateway-production-index)
Published: 2026-05-12
The state of production AI in 2026. Data from seven months of AI Gateway traffic across hundreds of models and tens of trillions of tokens.

### [How Superset built the IDE for AI agents on Vercel](/customers/how-superset-built-the-ide-for-ai-agents-on-vercel)
Published: 2026-05-10
Category: Customer stories
  How Superset built the IDE for AI coding agents on Vercel, running up to 10 coding agents in parallel per developer and nearly 600 preview deployments a day.

### [How KIKO Milano scales for Black Friday](/customers/how-kiko-milano-scales-for-black-friday)
Published: 2026-05-05
Category: Customer stories
Global beauty brand KIKO Milano migrated from AWS to Vercel, eliminating Black Friday prep, cutting build times 75%, and accelerating their release cycle.

### [How General Intelligence used agents to build an agent platform on Vercel](/customers/how-general-intelligence-used-agents-to-build-an-agent-platform-on-vercel)
Published: 2026-05-04
Category: Customer stories
Learn how General Intelligence built Cofounder, a multi-tenant platform that gives founders an AI team, using their own coding agents on Vercel.

### [Introducing deepsec: The security harness for finding vulnerabilities in your codebase](/blog/introducing-deepsec-find-and-fix-vulnerabilities-in-your-code-base)
Published: 2026-05-04
Today we're open sourcing deepsec, an AI security harness that runs on your infrastructure, with your keys, against your code.

### [How GitBook serves 30,000 sites with sub-second content updates](/customers/how-gitbook-serves-30000-sites-with-sub-second-content-updates)
Published: 2026-05-01
Category: Customer stories
GitBook hosts 30,000 documentation sites on Vercel, serving 120 million monthly page views for companies like Nvidia, Zoom, and n8n. 

### [2026 Vercel AI Accelerator recap](/blog/2026-vercel-ai-accelerator-recap)
Published: 2026-04-28
Read the recap of the 2026 Vercel AI Accelerator, where 39 AI startups spent six weeks building with Vercel before pitching at Demo Day at our San Francisco headquarters.

### [How Zo Computer improved AI reliability 20x on Vercel](/customers/how-zo-computer-improved-ai-reliability-20x-on-vercel)
Published: 2026-04-17
Category: Customer stories
See how Zo Computer used Vercel AI Gateway and AI SDK to cut retry rates 20x, raise chat success to 99.93%, reduce P99 latency by 38%, and add new model support in under a minute while scaling its personal AI cloud platform.⁠

### [A new programming model for durable execution](/blog/a-new-programming-model-for-durable-execution)
Published: 2026-04-16
Vercel Workflows is now GA. Write durable, long-running functions in TypeScript or Python. No orchestrator, no Kubernetes, no separate infrastructure. 100M+ runs in beta across 1,500+ customers.

### [Agentic Infrastructure](/blog/agentic-infrastructure)
Published: 2026-04-09
The shift to agentic infrastructure. For fifty years, infrastructure assumed a human operator. Someone to configure the server, click the deploy button, or read the logs.

### [Zero Data Retention on AI Gateway](/blog/zdr-on-ai-gateway)
Published: 2026-04-08
Enforce zero data retention across your entire team and prevent providers from training on your data. AI Gateway handles routing and provider agreements for you.

### [Optimizing Vercel Sandbox snapshots](/blog/optimizing-vercel-sandbox-snapshots)
Published: 2026-04-02
Vercel Sandbox snapshots let you save and restore your entire filesystem. Learn how we optimized snapshot restores with parallel downloads, streaming decompression, and local NVMe caching.

### [How Waldium made a blog platform work for humans and AI alike](/customers/how-waldium-made-a-blog-platform-work-for-humans-and-ai-alike)
Published: 2026-04-01
Category: Customer stories
How Waldium made a blog platform work for humans and AI alike. Waldium started the way most content platforms do: building blogs for humans to read. But something Amrutha kept noticing was quietly changing who, and what, showed up to read them.

### [How FLORA shipped a creative agent on Vercel's AI stack](/customers/how-flora-shipped-a-creative-agent-on-vercels-ai-stack)
Published: 2026-03-31
Category: Customer stories
Flora’s FAUNA creative agent turns ideas into visual workflows on a digital canvas. Built on the Vercel AI Stack (AI SDK + Workflow SDK DurableAgent + Fluid Compute) to ship faster and iterate at scale.

### [Agent responsibly](/blog/agent-responsibly)
Published: 2026-03-30
There's a difference between leveraging AI and relying on it. A framework for shipping agent-generated code with the judgment and guardrails it requires.

### [Making Turborepo 96% faster with agents, sandboxes, and humans](/blog/making-turborepo-ninety-six-percent-faster-with-agents-sandboxes-and-humans)
Published: 2026-03-30
Category: Field Engineering
Turborepo 2.9 is up to 96% faster than Turborepo 2.8. Here's how we did it, using coding agents, a human in the loop, and Vercel Sandboxes.

### [Unified reporting for all AI Gateway usage](/blog/unified-reporting-for-your-ai-spend)
Published: 2026-03-25
Break down AI inference costs by model, provider, user, and pricing tier. The Custom Reporting API gives you the data to calculate margins, track customer usage, and optimize spend across BYOK and system credentials.

### [new.website joins forces with v0](/blog/new-website-joins-forces-with-v0)
Published: 2026-03-23
Category: v0
v0 and new.website have joined forces to shorten the path from prototype to production for websites and web apps.

### [SERHANT.'s playbook for rapid AI iteration](/customers/serhants-playbook-for-rapid-ai-iteration)
Published: 2026-03-23
Category: Customer stories
Learn how SERHANT. scaled its AI platform S.MPLE to 900+ real estate agents using Next.js, Vercel, and AI SDK, without replatforming or locking into a single model provider.

### [Build knowledge agents without embeddings](/blog/build-knowledge-agents-without-embeddings)
Published: 2026-03-19
Category: Field Engineering
Open source file-system and knowledge based agent template. Build AI agents that stay up to date with your knowledge base. Grep, find, and cat across your sources, no embeddings, no vector DB. 

### [Two startups at global scale without DevOps](/customers/two-startups-at-global-scale-without-devops)
Published: 2026-03-19
Category: Customer stories
Discover how small AI teams in APJ scale to millions of users without hiring platform engineers. See how Vercel powers their growth.

### [Chat SDK brings agents to your users](/blog/chat-sdk-brings-agents-to-your-users)
Published: 2026-03-19
Chat SDK is a unified TypeScript library for building chat bots that work across Slack, Discord, Teams, and more from a single codebase. Write once, deploy everywhere.

### [360 billion tokens, 3 million customers, 6 engineers](/customers/360-billion-tokens-3-million-customers-6-engineers)
Published: 2026-03-18
Category: Customer stories
Durable rewrote its multi-tenant AI platform onto Vercel to serve 3M customers with six engineers, cutting infra costs 3-4x and shipping new production agents in a single day.

### [Vercel Open Source Program: Winter 2026 cohort](/blog/vercel-open-source-program-winter-2026-cohort)
Published: 2026-03-17
Category: Community
Announcing the winter 2026 cohort of Vercel's Open Source Program. Open source community frameworks, libraries, and tools we rely on every day to build the web.

### [Meet the 2026 Vercel AI Accelerator Cohort](/blog/2026-vercel-ai-accelerator-cohort)
Published: 2026-03-16
The Vercel AI Accelerator is back with 39 early-stage teams from around the world, $8M+ in partner credits, and a Demo Day in San Francisco on April 16.

### [How Notion Workers run untrusted code at scale with Vercel Sandbox](/customers/notion-workers-vercel-sandbox)
Published: 2026-03-12
Category: Customer stories
Learn how Notion Workers uses Vercel Sandbox to run untrusted code at scale with hard VM isolation, credential injection, and dynamic network policies.

### [How we run Vercel's CDN in front of Discourse](/blog/how-we-run-vercels-cdn-in-front-of-discourse)
Published: 2026-03-10
Category: Security
Vercel Community uses Vercel as a CDN in front of monolithic servers like Discourse or Wordpress. Use microfrontends to incrementally migrate to a new app.

### [From idea to secure checkout in minutes with Stripe](/blog/from-idea-to-secure-checkout-in-minutes-with-stripe)
Published: 2026-03-05
You can now connect or import an existing Stripe account directly into your Vercel project, automatically configure environment variables, and move from Stripe Sandbox to production without manual key exchanges. 

### [Building Slack agents can be easy](/blog/building-slack-agents-can-be-easy)
Published: 2026-03-03
Build and deploy Slack agents in a single session. This skill handles configuration, secrets, and deployment, taking you from idea to production with just a conversation with your coding agent.

### [Scaling redirects to infinity on Vercel](/blog/scaling-redirects-to-infinity-on-vercel)
Published: 2026-03-03
Redirects are trivial at a small scale, but at millions, latency and cost become real systems problems. This is the story of how Vercel implemented bulk redirects.

### [Advancing Python typing](/blog/advancing-python-typing)
Published: 2026-03-02
PEP 827 introduces type manipulation to Python, bringing programmable, TypeScript-inspired type features to improve static typing, metaprogramming, and framework ergonomics.

### [Gamma builds design-first agents with Vercel](/customers/gamma-builds-design-first-agents-with-vercel)
Published: 2026-02-28
Category: Customer stories
How Gamma builds design-first AI agents at scale using the AI SDK, Vercel Functions, and the Edge Network.

### [How Avalara turns pipe dreams into patent-pending with v0 ](/customers/How-avalara-turns-pipedreams-into-patent-pending-with-v0)
Published: 2026-02-28
Category: Customer stories
For Chief Strategy and Product Officer Jayme Fishman, the path to modernizing Avalara starts with how it builds.

### [Keeping community human while scaling with agents](/blog/keeping-community-human-while-scaling-with-agents)
Published: 2026-02-27
Category: Community
Learn how Vercel scales community support with AI agents. We automated logistics to reclaim human focus, empowering our team to solve complex problems

### [How OpenEvidence built a healthcare AI that physicians actually trust](/customers/how-openevidence-built-a-healthcare-ai-that-physicians-can-trust)
Published: 2026-02-25
Category: Customer stories
How OpenEvidence built a healthcare AI that physicians actually trust and became the most widely used clinical decision support platform among U.S. clinicians, supporting over 20 million clinical consultations in January 2026. 

### [Security boundaries in agentic architectures](/blog/security-boundaries-in-agentic-architectures)
Published: 2026-02-24
Category: Security
A framework for drawing security boundaries in agentic architectures. Most agents run with zero isolation between the agent and the code it generates. Learn where to draw the boundaries, from secret injection to full application sandboxing.

### [Skills Night: 69,000+ ways agents are getting smarter](/blog/skills-night-69000-ways-agents-are-getting-smarter)
Published: 2026-02-20
Andrew Qu reflects on Skills Night SF: how a weekend project became 69,000 community-created skills, the security partnerships protecting them, and what eight partner demos revealed about agents, context, and the future of development.

### [Video Generation with AI Gateway](/blog/video-generation-with-ai-gateway)
Published: 2026-02-19
Build video generation into your apps with AI Gateway. Create product videos, dynamic content, and marketing assets at scale.

### [We Ralph Wiggumed WebStreams to make them 10x faster](/blog/we-ralph-wiggumed-webstreams-to-make-them-10x-faster)
Published: 2026-02-18
Category: Field Engineering
WebStreams had too much overhead on the server. We built a faster implementation. See how we achieved 10-14x gains in Next.js rendering benchmarks.

### [How Ramp kept 100% uptime through 100x traffic surges on Vercel ](/customers/how-ramp-kept-100-uptime-through-100x-traffic-surges-on-vercel)
Published: 2026-02-17
Category: Customer stories
Ramp rebuilt its marketing site on Next.js and Vercel, holding 100% uptime and 0% error rates through 100x traffic surges during stunt-marketing events.

### [How Stably ships AI testing agents in hours, not weeks](/customers/how-stably-ships-ai-testing-agents-in-hours-not-weeks)
Published: 2026-02-17
Category: Customer stories
How Stably, a 6-person team, ships AI testing agents faster with Vercel, moving from weeks to hours. Their shift highlights how Vercel's platform eliminates infrastructure anxiety, boosting autonomous testing and enabling quick enterprise growth.

### [How we built AEO tracking for coding agents](/blog/how-we-built-aeo-tracking-for-coding-agents)
Published: 2026-02-09
Learn how we built an AI Engine Optimization system to track coding agents using Vercel Sandbox, AI Gateway, and Workflows for isolated execution.

### [Anyone can build agents, but it takes a platform to run them](/blog/anyone-can-build-agents-but-it-takes-a-platform-to-run-them)
Published: 2026-02-09
Why competitive advantage in AI comes from the platform you deploy agents on, not the agents themselves.

### [Introducing Geist Pixel](/blog/introducing-geist-pixel)
Published: 2026-02-06
Geist Pixel is a bitmap-inspired typeface built on the same foundations as Geist and Geist Mono, reinterpreted through a strict pixel grid. It’s precise, intentional, and unapologetically digital.

### [The Vercel AI Accelerator is back with $6m in credits](/blog/the-vercel-ai-accelerator-is-back-with-6-million-in-credits)
Published: 2026-02-05
Category: Company News
A six-week program to help you scale your AI company offering over $6M in credits from Vercel, v0, AWS, and leading AI platforms

### [Making agent-friendly pages with content negotiation](/blog/making-agent-friendly-pages-with-content-negotiation)
Published: 2026-02-03
Category: Field Engineering
Learn how Vercel uses HTTP content negotiation to serve markdown to agents and HTML to humans from the same URL, reducing response sizes by 90% while keeping both versions synchronized.

### [The Vercel OSS Bug Bounty program is now available](/blog/the-vercel-oss-bug-bounty-program-is-now-available)
Published: 2026-02-03
Category: Security
Vercel is opening its open source software bug bounty program to the public for researchers find vulnerabilities and make OSS safer

### [Introducing the new v0](/blog/introducing-the-new-v0)
Published: 2026-02-03
Category: v0
The new v0 brings production-ready AI coding to enterprises with git workflows, security, and real integrations. Ship faster with agents and teams.

### [Run untrusted code with Vercel Sandbox, now generally available](/blog/vercel-sandbox-is-now-generally-available)
Published: 2026-01-30
AI agents need secure, isolated environments that spin up instantly. Vercel Sandbox is now generally available with filesystem snapshots, container support, and production reliability.

### [How Stripe built a game-changing app in a single flight with v0](/customers/how-stripe-built-a-game-changing-app-in-a-single-flight-with-v0)
Published: 2026-01-28
Category: Customer stories
Stripe built a production GTM value calculator in a single flight using v0, boosting adoption 288% and cutting value analysis time by 80%.

### [How Sensay went from zero to product in six weeks](/customers/how-sensay-went-from-zero-to-product-in-six-weeks)
Published: 2026-01-27
Category: Customer stories
Sensay went from zero to an MVP launch in six weeks for Web Summit. With Vercel preview deployments, feature flags, and rollbacks, the team shipped fast without a DevOps team.

### [AGENTS.md outperforms skills in our agent evals](/blog/agents-md-outperforms-skills-in-our-agent-evals)
Published: 2026-01-27
A compressed 8KB docs index in AGENTS.md achieved 100% on Next.js 16 API evals. Skills maxed at 79%. Here's what we learned and how to set it up.

### [Agent skills explained: An FAQ](/blog/agent-skills-explained-an-faq)
Published: 2026-01-26
A plainspoken Skills FAQ with a ready-to-use guide: what skill packages are, how agents load them, what skills-ai.dev is, how Skills compare to MCP, plus security and alternatives.

### [Testing if "bash is all you need"](/blog/testing-if-bash-is-all-you-need)
Published: 2026-01-22
We tested bash vs SQL agents on structured data queries. SQL dominated, but combining both tools achieved 100% accuracy. Try the open-source eval harness.

### [AWS databases are now live on the Vercel Marketplace and v0](/blog/aws-databases-are-now-live-on-the-vercel-marketplace-and-v0)
Published: 2026-01-15
v0 can now provision AWS databases as it builds your app. Aurora PostgreSQL, DynamoDB, and Aurora DSQL available in the Vercel Marketplace.

### [Use Perplexity Web Search with Vercel AI Gateway](/blog/use-perplexity-web-search-with-vercel-ai-gateway)
Published: 2026-01-14
Vercel AI Gateway now supports Perplexity Web Search as a model-agnostic tool that works for all models and providers. Use the tool to get access to the most recent information to supplement your AI queries.

### [Introducing: React Best Practices](/blog/introducing-react-best-practices)
Published: 2026-01-14
Category: Field Engineering
We've encapsulated 10+ years of React and Next.js optimization knowledge into react-best-practices, a structured repository optimized for AI agents and LLMs. 

### [Nick Bogaty joins Vercel as Chief Revenue Officer](/blog/nick-bogaty-joins-vercel-as-chief-revenue-officer)
Published: 2026-01-13
Nick Bogaty joins Vercel as first Chief Revenue Officer from Adobe, bringing more than 20 years of enterprise GTM experience to lead an AI-forward sales organization.

### [How Mux shipped durable video workflows with their @mux/ai SDK](/blog/how-mux-shipped-durable-video-workflows-with-their-mux-ai-sdk)
Published: 2026-01-12
Learn how Mux built durable AI video workflows into their @mux/ai SDK using Workflow DevKit, with automatic retries, state persistence, and zero infrastructure setup.

### [How to build agents with filesystems and bash](/blog/how-to-build-agents-with-filesystems-and-bash)
Published: 2026-01-09
How to build agents with filesystems and bash, production agents, context management, template, bash

### [How we made v0 an effective coding agent ](/blog/how-we-made-v0-an-effective-coding-agent)
Published: 2026-01-07
Category: v0
v0’s composite AI pipeline boosts reliability by fixing errors in real time. Learn how dynamic system prompts, LLM Suspense, and autofixers work together to deliver stable, working web app generations at scale.

### [Stopping the slow death of internal tools](/blog/stopping-the-slow-death-of-internal-tools)
Published: 2025-12-27
Category: v0
Internal tools often decay due to high maintenance costs and security tradeoffs. Learn how Vercel uses v0 to build secure, sustainable custom software that business teams can ship and maintain without pulling engineers off the roadmap.

### [Pixel Portraits: AI generated trading cards](/blog/pixel-portraits-ai-generated-trading-cards)
Published: 2025-12-23
How Vercel built AI-generated pixel trading cards for Next.js Conf and Ship AI, then turned the same pipeline into a v0 template and festive holiday experiment.

### [We removed 80% of our agent’s tools](/blog/we-removed-80-percent-of-our-agents-tools)
Published: 2025-12-22
We spent months building a sophisticated text-to-sql agent, but as it turns out, sometimes simpler is better. Giving it the ability to execute arbitrary bash commands outperformed everything we built. We call this a file system agent.

### [AI SDK 6](/blog/ai-sdk-6)
Published: 2025-12-22
Category: Field Engineering
Introducing agents, tool execution approval, DevTools, full MCP support, reranking, image editing, and more.

### [Our $1 million hacker challenge for React2Shell](/blog/our-million-dollar-hacker-challenge-for-react2shell)
Published: 2025-12-19
Category: Security
We paid $1M to security researchers to break our WAF. Here's what we learned defending against React2Shell.

### [Cline now runs on Vercel AI Gateway](/blog/cline-on-ai-gateway)
Published: 2025-12-16
Cline scales its open source coding agent with Vercel AI Gateway, delivering global performance, transparent pricing, and enterprise reliability.

### [How to prompt v0](/blog/how-to-prompt-v0)
Published: 2025-12-15
Category: v0
The best v0 prompts include three things. What you're building, who uses it and when, and your design constraints. This guide walks through the framework with side-by-side tests showing faster generation times, less code, and better UX decisions.

### [Build smarter workflows with Notion and v0](/blog/build-smarter-workflows-with-notion-and-v0)
Published: 2025-12-15
Category: v0
v0 now connects to Notion via MCP. Build dashboards, tools, and prototypes from your existing docs and databases, and write generated content back to your workspace.

### [Vercel launches partner certification](/blog/vercel-launches-partner-certification)
Published: 2025-12-10
Vercel introduces the inaugural cohort of Vercel Certified Solution Partners. These industry-leading teams share our commitment to creating a faster, more accessible, and innovative web for our customers.﻿

### [Inside Workflow DevKit: How framework integrations work](/blog/inside-workflow-devkit-how-framework-integrations-work)
Published: 2025-12-09
Category: Field Engineering
A deep dive into how Workflow DevKit integrates with modern frameworks, from Next.js and SvelteKit to Express and Hono, using a unified pattern for creating framework integrations.

### [React2Shell Security Bulletin](https://vercel.com/kb/bulletin/react2shell)
Published: 2025-12-05
Active exploits exist for CVE-2025-55182 (React2Shell). Vercel WAF protections are in place, but upgrading is the only complete fix. Detection and mitigation steps for Next.js and React Server Components.

### [Billions of requests: Black Friday-Cyber Monday 2025](/blog/bfcm-2025)
Published: 2025-12-02
Every year, Black Friday and Cyber Monday reveal how people shop, browse, and discover products at global scale. For Vercel, the weekend is not a different operating mode. The platform behaves the same way it does every day, only at heightened scale.

### [Investing in the Python ecosystem](/blog/investing-in-the-python-ecosystem)
Published: 2025-12-02
Vercel welcomes the Gel Data team and deepens support for the Python ecosystem through PSF sponsorships, community funding, and improved Python developer tools.

### [AWS Databases coming to the Vercel Marketplace](/blog/aws-databases-coming-to-the-vercel-marketplace)
Published: 2025-12-01
Vercel Marketplace adds Aurora PostgreSQL, Amazon DynamoDB, Aurora DSQL, available December 15. One-click provisioning, zero-config setup, and full v0 support.

### [How we built the v0 iOS app](/blog/how-we-built-the-v0-ios-app)
Published: 2025-11-24
Category: v0
The v0 engineering team breaks down the challenges and decisions behind building the v0 app for iOS.

### [Security through design: Creating the improved Firewall experience](/blog/security-through-design-creating-the-improved-firewall-experience)
Published: 2025-11-24
Category: Security
Vercel introduces a new Firewall UI for better surfacing of events and alerts while providing deeper information on mitigation activity

### [Workflow Builder: Build your own workflow automation platform](/blog/workflow-builder-build-your-own-workflow-automation-platform)
Published: 2025-11-24
Category: Field Engineering
Workflow Builder is a free, open-source template, powered by Next.js and the Workflow DevKit, that helps you build workflow builder applications and agents.

### [Vercel Open Source Program: Fall 2025 cohort](/blog/vercel-open-source-program-fall-2025-cohort)
Published: 2025-11-21
Category: Community
Announcing the fall 2025 cohort of Vercel's Open Source Program. Open source community frameworks, libraries, and tools we rely on every day to build the web.

### [Self-driving infrastructure](/blog/self-driving-infrastructure)
Published: 2025-11-21
At Vercel, we’re building self-driving infrastructure, a system that autonomously manages production operations, improves application code using real-world insights, and learns from the unpredictable nature of production itself.

### [Vercel collaborates with Google for Gemini 3 Pro Preview launch](/blog/vercel-collaborates-with-google-for-gemini-3-pro-launch)
Published: 2025-11-18
The Gemini 3 Pro Preview model, released today, is now available through AI Gateway and in production on v0.app.

### [Vercel: The anti-vendor-lock-in cloud](/blog/vercel-the-anti-vendor-lock-in-cloud)
Published: 2025-11-10
Framework-defined infrastructure interprets your code and provisions what you need, keeping your application portable across any platform.

### [How Nous Research used BotID to block automated abuse at scale](/customers/how-nous-research-used-botid-to-block-automated-abuse-at-scale)
Published: 2025-11-07
Category: Customer stories
Vercel BotID Deep Analysis protected Nous Research by blocking advanced automated abuse from attacking their application

### [How AI Gateway runs on Fluid compute](/blog/how-ai-gateway-runs-on-fluid-compute)
Published: 2025-11-06
The AI Gateway is a simple application deployed on Vercel, but it achieves scale, efficiency, and resilience by running on Fluid compute and leveraging Vercel’s global infrastructure.

### [What we learned building agents at Vercel](/blog/what-we-learned-building-agents-at-vercel)
Published: 2025-11-06
We're presenting a simple methodology for discovering successful agent projects that perform well with current generation AI

### [Build and deploy data applications on Snowflake with v0](/blog/build-and-deploy-data-applications-on-snowflake-with-v0)
Published: 2025-11-04
Category: v0
The v0 Snowflake integration lets you build and deploy Next.js data applications with natural language. Your data stays secure in your Snowflake account.

### [BotID Deep Analysis catches a sophisticated bot network in real-time](/blog/botid-deep-analysis-catches-a-sophisticated-bot-network-in-real-time)
Published: 2025-10-31
Category: Field Engineering
BotID Deep Analysis is a sophisticated, invisible bot detection product. This article is about how BotID Deep Analysis adapted to a novel attack in real time, and successfully classified sessions that would have slipped through other services.

### [Vercel Agent can now run AI investigations](/blog/vercel-agent-can-now-run-ai-investigations)
Published: 2025-10-31
Vercel Agent Investigation intelligently conducts incident response investigations to alert, analyze, and suggest remediation steps

### [Vercel achieves TISAX AL2 compliance to serve automotive partners](/blog/vercel-achieves-tisax-al2-compliance-to-serve-automotive-partners)
Published: 2025-10-29
Category: Security
Vercel has achieved TISAX Assessment Level 2 security standard to align with automotive and manufacturing industries

### [Bun runtime on Vercel Functions](/blog/bun-runtime-on-vercel-functions)
Published: 2025-10-28
Category: Field Engineering
Vercel Functions now supports the Bun runtime, giving developers faster performance options and greater flexibility for optimizing JavaScript workloads.

### [David Totten Joins Vercel to Lead Global Field Engineering](/blog/david-totten-joins-vercel-to-lead-global-field-engineering)
Published: 2025-10-27
David Totten joins Vercel as VP of Global Field Engineering from Databricks to oversee Sales Engineering, Developer Success, Professional Services, and Customer Support Engineering under one integrated organization

### [Vercel Ship AI 2025 recap](/blog/ship-ai-2025-recap)
Published: 2025-10-27
Earlier this year we introduced the foundations of the AI Cloud: a platform for building intelligent systems that think, plan, and act. At Ship AI, we showed what comes next. What and how to build with the AI Cloud. 

### [You can just ship agents](/blog/you-can-just-ship-agents)
Published: 2025-10-23
Vercel AI Cloud combines unified model routing and failover, elastic cost-efficient compute that only bills for active CPU time, isolated execution for untrusted code, and workflow durability that survives restarts, deploys, and long pauses.

### [AI agents and services on the Vercel Marketplace](/blog/ai-agents-and-services-on-the-vercel-marketplace)
Published: 2025-10-23
Category: Company News
Agents and Tools are available in the Vercel Marketplace, enabling AI-powered workflows in your projects with native integrations, unified billing, and built-in observability.

### [Built-in durability: Introducing Workflow Development Kit](/blog/introducing-workflow)
Published: 2025-10-23
The Workflow Development Kit (WDK) makes async workflows in TypeScript reliable, durable, fault-tolerant, and portable across any cloud.

### [Zero-config backends on Vercel AI Cloud](/blog/zero-config-backends-on-vercel-ai-cloud)
Published: 2025-10-23
Build, scale, and orchestrate AI backends on Vercel. Deploy Python or Node frameworks with zero config and optimized compute for agents and workflows.

### [Introducing Vercel Agent: Your new Vercel teammate](/blog/introducing-vercel-agent)
Published: 2025-10-23
Vercel Agent provides AI-powered code reviews and production investigations, delivering accurate, context-aware insights to help you ship reliable software.

### [Update regarding Vercel service disruption on October 20, 2025](/blog/update-regarding-vercel-service-disruption-on-october-20-2025)
Published: 2025-10-21
Update regarding Vercel service disruption on October 20, 2025. Read the summary of impact, timeline, root cause, and steps we’re taking to improve reliability.

### [Agents at work, a partnership with Salesforce and Slack](/blog/agents-at-work-a-partnership-with-salesforce-and-slack)
Published: 2025-10-15
Category: Company News
Vercel and Salesforce are partnering to help teams build, ship, and scale AI agents across the Salesforce ecosystem, starting with Slack. 

### [Running Next.js inside ChatGPT: A deep dive into native app integration](/blog/running-next-js-inside-chatgpt-a-deep-dive-into-native-app-integration)
Published: 2025-10-15
Next.js now runs natively in ChatGPT with working navigation, React Server Components, and full features. Learn how we made this possible behind ChatGPTs triple iframe architecture and deploy our starter template to get started.

### [Talha Tariq joins Vercel as CTO of Security](/blog/talha-tariq-joins-vercel-as-cto-security)
Published: 2025-10-15
Talha Tariq joins Vercel as CTO of Security, bringing expertise from HashiCorp and IBM to lead security innovation in the AI era.

### [Just another (Black) Friday](/blog/just-another-black-friday)
Published: 2025-10-15
Vercel customers can treat Black Friday like just another day, ready to scale to billions of requests.

### [Server rendering benchmarks: Fluid Compute and Cloudflare Workers](/blog/fluid-compute-benchmark-results)
Published: 2025-10-09
Fluid Compute outperforms Cloudflare Workers by 1.2x–5x in server-side rendering benchmarks, offering faster, more consistent response times through an optimized in-region architecture.

### [Towards the AI Cloud: Our Series F](/blog/series-f)
Published: 2025-09-30
Category: Company News
Today, Vercel announced an important milestone: a Series F funding round valuing our company at $9.3 billion.

### [Collaborating with Anthropic on Claude Sonnet 4.5 to power intelligent coding agents](/blog/collaborating-with-anthropic-on-claude-sonnet-4-5)
Published: 2025-09-29
Claude Sonnet 4.5 is now available on Vercel AI Gateway and across the Vercel AI Cloud. Also introducing a new coding agent platform template.

### [Preventing the stampede: Request collapsing in the Vercel CDN ](/blog/cdn-request-collapsing)
Published: 2025-09-25
Category: Field Engineering
The Vercel CDN now supports request collapsing for ISR routes. For a given path, only one function invocation per region runs at once, no matter how many concurrent requests arrive.

### [BotID uncovers hidden SEO poisoning](/blog/botid-uncovers-hidden-seo-poisoning)
Published: 2025-09-22
A financial institution's suspicious bot traffic turned out to be Google bots crawling SEO-poisoned URLs from years ago. Here's how BotID revealed the real problem.

### [How we made global routing faster with Bloom filters](/blog/how-we-made-global-routing-faster-with-bloom-filters)
Published: 2025-09-19
Category: Field Engineering
We replaced slow JSON path lookups with Bloom filters in our global routing service, cutting memory usage by 15% and reducing 99th percentile lookup times from hundreds of milliseconds to under 1 ms. Here’s how we did it.

### [What you need to know about vibe coding](/blog/what-you-need-to-know-about-vibe-coding)
Published: 2025-09-18
Category: v0
Vibe coding is revolutionizing how we work. English is now the fastest growing programming language. Our state of vibe coding report outlines what you need to know. 

### [Scale to one: How Fluid solves cold starts](/blog/scale-to-one-how-fluid-solves-cold-starts)
Published: 2025-09-18
Learn how Vercel solves serverless cold starts with scale to one, Fluid compute, predictive scaling, and caching to keep functions warm and fast.

### [Addressing security and quality issues with MCP tools in AI Agent](/blog/generate-static-ai-sdk-tools-from-mcp-servers-with-mcp-to-ai-sdk)
Published: 2025-09-17
Use mcp-to-ai-sdk to generate MCP tools directly into your project. Gain security, reliability, and prompt-tuned control while avoiding dynamic MCP risks.

### [AI agents at scale: Rox’s Vercel-powered revenue operating system](/customers/ai-agents-at-scale-roxs-vercel-powered-revenue-operating-system)
Published: 2025-09-16
Category: Customer stories
Learn more about how Rox runs global, AI-driven sales ops on fast, reliable infrastructure thanks to Vercel

### [Helly Hansen migrated to Vercel and drove 80% Black Friday growth ](/customers/how-helly-hansen-migrated-to-vercel-and-drove-80-black-friday-growth)
Published: 2025-09-15
Category: Customer stories
The 150-year-old Norwegian brand leveraged Next.js and Vercel to achieve 154% Black Friday growth and 30%+ conversion lift while competing against industry titans in a crowded space

### [Introducing Vercel Drains: Complete observability data, anywhere](/blog/introducing-vercel-drains)
Published: 2025-09-15
Vercel Drains give you a single way to stream observability data out of Vercel and into the systems your team already rely on.

### [Introducing x402-mcp: Open protocol payments for MCP tools](/blog/introducing-x402-mcp-open-protocol-payments-for-mcp-tools)
Published: 2025-09-12
We built x402-mcp to integrate x402 payments with Model Context Protocol (MCP) servers and the Vercel AI SDK.

### [MongoDB Atlas is now available on the Vercel Marketplace](/blog/mongodb-atlas-is-now-available-on-the-vercel-marketplace)
Published: 2025-09-10
MongoDB Atlas is now available on the Vercel Marketplace, enabling developers to provision, manage, and scale fully managed MongoDB databases directly from the Vercel dashboard.

### [The second wave of MCP: Building for LLMs, not developers](/blog/the-second-wave-of-mcp-building-for-llms-not-developers)
Published: 2025-09-09
Category: Field Engineering
The second wave of MCP, building for LLMs, not developers. Explore the evolution of MCP as it shifts from developer-focused tools to LLM-native integrations. Discover the future of AI connectivity.

### [A more flexible Pro plan for modern teams](/blog/new-pro-pricing-plan)
Published: 2025-09-09
Category: Company News
We’re updating Vercel’s Pro plan to better align with how modern teams collaborate and how applications consume infrastructure, and how workloads are changing shape with AI.

### [Critical npm supply chain attack response – September 8, 2025](/blog/critical-npm-supply-chain-attack-response-september-8-2025)
Published: 2025-09-08
Category: Security
How Vercel responded to the September 2025 npm supply chain attack on chalk, debug and 16 other packages. Incident timeline, impact analysis, and customer remediation.

### [Stress testing Biome's noFloatingPromises lint rule](/blog/stress-testing-biomes-nofloatingpromises-lint-rule)
Published: 2025-09-04
Category: Field Engineering
We partnered with Biome to push their noFloatingPromises lint rule to the limit, uncovering edge cases and showing how we solve hard problems together.

### [Open SDK strategy](/blog/open-sdk-strategy)
Published: 2025-09-03
Vercel’s Open SDK strategy commits to building frameworks, SDKs, and tools in the open, under permissive licenses. Learn how we’re avoiding lock-in, ensuring portability, and investing in open source to build a better web for everyone.

### [Preparing for the worst: Our core database failover test](/blog/preparing-for-the-worst-our-core-database-failover-test)
Published: 2025-08-28
Category: Field Engineering
On July 24, 2025, we successfully performed a full production failover of our core control-plane database from Azure West US to East US 2 with zero customer impact.

### [AI-powered prototyping with design systems](/blog/ai-powered-prototyping-with-design-systems)
Published: 2025-08-22
Category: v0
Why AI-native design systems unlock true brand-ready, production-aligned prototyping for teams using v0

### [AI Gateway: Production-ready reliability for your AI apps](/blog/ai-gateway-is-now-generally-available)
Published: 2025-08-21
AI Gateway, now generally available, ensures availability when a provider fails, avoiding low rate limits and providing consistent reliability for AI workloads.

### [Rethinking prototyping, requirements, and project delivery at Code and Theory](/customers/rethinking-prototyping-requirements-and-project-delivery-at-code-and-theory)
Published: 2025-08-20
Category: Customer stories
How the agency, Code and Theory, cuts time-to-prototype by 75% and moves faster from idea to execution

### [<script type="text/llms.txt">](/blog/a-proposal-for-inline-llm-instructions-in-html)
Published: 2025-08-20
llms.txt is an emerging standard for making content such as docs available for direct consumption by AIs. We’re proposing a convention to include such content directly in HTML responses.

### [If agents are building your app, who gets the W-2?](/blog/if-agents-are-building-your-app-who-gets-the-w-2)
Published: 2025-08-18
If agents can design, build, test, and deploy features, their work should be treated like a developer's under GAAP. With modern AI logging, you can tie usage directly to capitalizable development activity.

### [The three types of AI bot traffic and how to handle them](/blog/the-three-types-of-ai-bot-traffic-and-how-to-handle-them)
Published: 2025-08-13
Category: Security
Bots account for over 20% of all web traffic. A quarter of that is AI crawlers alone. But not all bots are bad. Some are key to how your site gets discovered. Learn how AI bots shape visibility and what happens if you block them.

### [The real serverless compute to database connection problem, solved](/blog/the-real-serverless-compute-to-database-connection-problem-solved)
Published: 2025-08-13
Serverless compute does not mean you need more database connections. The math is the same for serverful and serverless. The real difference is what happens when functions suspend. We solve this issue with Fluid compute.

### [How Coxwave delivers GenAI value faster with Vercel](/customers/how-coxwave-delivers-genai-value-faster-with-vercel)
Published: 2025-08-13
Category: Customer stories
Coxwave's journey to cutting deployment times by 85% and building AI-native products faster with Vercel

### [Cutting delivery times in half with v0](/customers/cutting-delivery-times-in-half-with-v0)
Published: 2025-08-12
Category: Customer stories
Learn how Ready.net uses v0 to reduce ambiguity and accelerate feedback loops with limited resources

### [v0.dev -> v0.app](/blog/v0-app)
Published: 2025-08-11
Category: v0
v0.dev is now v0.app, the AI builder for everyone: founders, designers, developers, marketers, sales, finance, and more 

### [How Zapier scales product partnerships with v0](/customers/how-zapier-scales-product-partnerships-with-v0)
Published: 2025-08-08
Category: Customer stories
The team behind Zapier’s embedded platform uses v0 to turn partner conversations into scalable integrations

### [Vercel collaborates with OpenAI for GPT-5 launch](/blog/vercel-collaborates-with-openai-for-gpt-5-launch)
Published: 2025-08-07
The GPT-5 family of models released today, are now available through AI Gateway and are in production on our own v0.dev applications. Thanks to OpenAI, Vercel has been testing these models for a few weeks in v0, Next.js, AI SDK, and Vercel Sandbox.

### [Vercel is the only vendor to be recognized as a Visionary in the 2025 Gartner® Magic Quadrant™ for Cloud-Native Application Platforms](/blog/gartner-mq-visionary-2025)
Published: 2025-08-07
Category: Company News
We’re honored to be the only vendor recognized as a Visionary in the 2025 Gartner® Magic Quadrant™ for Cloud Native Application Platforms.

### [Introducing Vercel MCP: Connect Vercel to your AI tools](/blog/introducing-vercel-mcp-connect-vercel-to-your-ai-tools)
Published: 2025-08-06
Vercel now has an official hosted MCP server (aka Vercel MCP), which you can use to connect your favorite AI tools, such as Claude or VS Code, directly to Vercel.

### [v0: vibe coding, securely](/blog/v0-vibe-coding-securely)
Published: 2025-08-04
Category: v0
Vibe coding makes it possible for anyone to ship a viral app. But every line of AI-generated code is a potential vulnerability. Security cannot be an afterthought, it must be the foundation. Turn ideas into secure apps with v0.

### [A new wave of software, shipped on Vercel](/blog/shipped-on-vercel)
Published: 2025-08-01
Category: Company News
We're launching a new way to showcase standout products built and shipped on Vercel. Submit your project. 

### [Vercel Open Source Program: Summer cohort](/blog/summer-2025-oss-program)
Published: 2025-07-31
Category: Community
Announcing the summer 2025 cohort of Vercel's Open Source Program. Open source community frameworks, libraries, and tools we rely on every day to build the web,

### [AI SDK 5](/blog/ai-sdk-5)
Published: 2025-07-31
Category: Field Engineering
Introducing type-safe chat, agentic loop control, new specification, tool enhancements,  speech generation, and more.

### [Join the v0 Ambassador Program](/blog/join-the-v0-ambassador-program)
Published: 2025-07-29
Category: v0
Apply today to join the v0 Ambassador Program and help others discover the magic of what's possible with v0. 

### [Fluid: How we built serverless servers](/blog/fluid-how-we-built-serverless-servers)
Published: 2025-07-28
Fluid Compute cuts cold starts and compute costs by up to 95%, scaling I/O-bound and AI workloads efficiently across 45B+ weekly requests.

### [Model Context Protocol (MCP) explained: An FAQ](/blog/model-context-protocol-mcp-explained)
Published: 2025-07-25
Model Context Protocol (MCP) is a new spec that helps standardize the way large language models (LLMs) access data and systems, extending what they can do beyond their training data. 

### [Vercel and Solara6 partner to build better ecommerce experiences](/blog/vercel-and-solara6-partner-to-build-better-ecommerce-experiences)
Published: 2025-07-25
Category: Company News
Solara6 is partnering with us to help ecommerce brands ship faster and deploy with confidence. Through this partnership, ecommerce teams working with Solara6 can expect improved SEO, site speed, and reliability during peak traffic moments.

### [Build your own AI app builder with the v0 Platform API](/blog/build-your-own-ai-app-builder-with-the-v0-platform-api)
Published: 2025-07-23
Category: v0
Learn how to build, extend, and automate AI-generated apps like BI tools and website builders with v0 Platform API

### [Grep a million GitHub repositories via MCP](/blog/grep-a-million-github-repositories-via-mcp)
Published: 2025-07-17
Search 1M+ GitHub repositories from your AI agent using Grep's MCP server. Your agent can now reference coding patterns and solutions used in open source projects to solve problems.

### [The AI Cloud: A unified platform for AI workloads](/blog/the-ai-cloud-a-unified-platform-for-ai-workloads)
Published: 2025-07-10
We made it simple to build, preview, and ship any frontend, from marketing pages to dynamic apps, without managing infrastructure. Now we’re introducing the next layer: the Vercel AI Cloud.

### [NuxtLabs joins Vercel](/blog/nuxtlabs-joins-vercel)
Published: 2025-07-08
Category: Company News
NuxtLabs, creators of Nuxt and Nitro, are joining Vercel. Same license, roadmap, and open governance, but now in a joint mission to build the best web.

### [Vercel Ship 2025 recap](/blog/vercel-ship-2025-recap)
Published: 2025-06-26
Category: Company News
Vercel Ship 2025 added new building blocks for an AI era: Fast, flexible, and secure by default. Lower costs with Fluid's Active CPU pricing, Rolling Releases for safer deployments, invisible CAPTCHA with BotID. See these and more in our recap.

### [​Introducing BotID, invisible bot filtering for critical routes](/blog/introducing-botid)
Published: 2025-06-25
BotID is a new invisible CAPTCHA layer of protection that stops sophisticated bots before they reach your backend. It's built to secure critical routes like checkouts, logins, and signups or actions that trigger expensive calls like LLM-powered APIs.

### [Introducing Active CPU pricing for Fluid compute](/blog/introducing-active-cpu-pricing-for-fluid-compute)
Published: 2025-06-25
Fluid compute now uses Active CPU pricing. Only pay CPU rates when your function is actively computing. Building on existing Fluid gains, this brings additional savings of up to 90% for workloads like LLM calls, AI agents, or tasks with idle time.

### [WPP and Vercel: Bringing AI to the creative process](/blog/wpp-and-vercel-bringing-ai-to-the-creative-process)
Published: 2025-06-24
Category: Company News
Announcing an expansion of our partnership with WPP, a first-of-its-kind agency collaboration that now brings v0 and AI SDK directly to WPP's global network of creative teams and their clients.

### [Keith Messick joins Vercel as CMO](/blog/keith-messick-joins-vercel-as-cmo)
Published: 2025-06-23
We’re welcoming Keith Messick as Chief Marketing Officer to support our growth, engage on more channels, and (as always) amplify the voice of the developer. Keith is a longtime enterprise CMO and comes to Vercel from database leader, Redis.

### [Tray.ai cut build times from a day to minutes with Vercel](/customers/tray-ai-cut-build-times-from-a-day-to-minutes-with-vercel)
Published: 2025-06-16
Category: Customer stories
Tray.ai cut build times from a full day to just two minutes after migrating to Vercel. By consolidating infrastructure and updating their tech stack, they now deliver over a million monthly page views with a faster, more resilient site.

### [Building efficient MCP servers](/blog/building-efficient-mcp-servers)
Published: 2025-06-12
Category: Field Engineering
MCP is becoming the standard for building AI model integrations. See how you can use Vercel's open-source MCP adapter to quickly build your own MCP server, like the teams at Zapier, Composio, and Solana.

### [Designing and building the Vercel Ship conference platform](/blog/designing-and-building-the-vercel-ship-conference-platform)
Published: 2025-06-11
Here's how we designed and built our Vercel Ship conference platform. We generated 15,000+ images and videos with tools like Flux, Veo 2, Runway, and Ideogram. Then, we moved to v0 for prototyping. See our iterations, examples, tech stack, and more.

### [How we’re adapting SEO for LLMs and AI search](/blog/how-were-adapting-seo-for-llms-and-ai-search)
Published: 2025-06-10
AI is changing how content gets discovered. Now, SEO ranking ≠ LLM visibility. No one has all the answers, but here's how we're adapting our approach to SEO for LLMs and AI search.

### [Building secure AI agents](/blog/building-secure-ai-agents)
Published: 2025-06-09
Category: Field Engineering
Learn how to design secure AI agents that resist prompt injection attacks. Understand tool scoping, input validation, and output sanitization strategies to protect LLM-powered systems.

### [The no-nonsense approach to AI agent development](/blog/the-no-nonsense-approach-to-ai-agent-development)
Published: 2025-06-04
Category: Field Engineering
Learn how to build reliable, domain-specific AI agents by simulating tasks manually, structuring logic with code, and optimizing with real-world feedback. A clear, hands-on approach to practical automation.

### [Introducing the v0 composite model family](/blog/v0-composite-model-family)
Published: 2025-06-01
Category: Field Engineering
Learn how v0's composite AI models combine RAG, frontier LLMs, and AutoFix to build accurate, up-to-date web app code with fewer errors and faster output.

### [Fluid compute: Evolving serverless for AI workloads](/blog/fluid-compute-evolving-serverless-for-ai-workloads)
Published: 2025-05-30
Fluid, our newly announced compute model, eliminates wasted compute by maximizing resource efficiency. Instead of launching a new function for every request, it intelligently reuses available capacity, ensuring that compute isn’t sitting idle.

### [Vercel security roundup: improved bot defenses, DoS mitigations, and insights](/blog/vercel-security-roundup-improved-bot-defenses-dos-mitigations-and-insights)
Published: 2025-05-23
Category: Security
Since February, Vercel blocked over 148 billion attacks from 108 million IPs. This roundup highlights improvements to bot protection, DoS mitigation, and firewall tooling to help teams build securely by default.

### [How Vapi built their MCP server on Vercel](/customers/vapi-mcp-server-on-vercel)
Published: 2025-05-21
Category: Customer stories
Vapi has used Vercel's MCP Adapter to deploy and host their MCP server on Vercel, leveraging the benefits of Fluid Compute

### [Vercel Blob is now generally available: Cost-efficient, durable storage](/blog/vercel-blob-now-generally-available)
Published: 2025-05-21
Vercel Blob is now generally available, providing durable object storage that's integrated with Vercel's application delivery network.

### [Introducing the AI Gateway](/blog/ai-gateway)
Published: 2025-05-20
Category: Company News
With the AI Gateway, build with any model instantly. No API keys, no configuration, no vendor lock-in.

### [How Fern delivers 6M+ monthly views and 80% faster docs with Vercel](/customers/how-fern-delivers-6m-monthly-views-and-80-faster-docs-with-vercel)
Published: 2025-05-15
Category: Customer stories
Fern used Vercel and Next.js to achieve efficient multi-tenancy, faster development cycles, and 50-80% faster load times

### [How Consensys rebuilt MetaMask.io with Vercel and Next.js](/customers/how-consensys-rebuilt-metamask-io-with-vercel-and-next-js)
Published: 2025-05-14
Category: Customer stories
Learn how Consensys modernized MetaMask.io using Vercel and Next.js—cutting deployment times, improving collaboration across teams, and unlocking dynamic content with serverless architecture.

### [Updated v0 pricing](/blog/updated-v0-pricing)
Published: 2025-05-13
Category: v0
More flexible pricing for v0 that scales with your usage and lets you pay on-demand through credits.

### [The spring 2025 cohort of Vercel’s Open Source Program](/blog/spring25-oss-program)
Published: 2025-05-12
Category: Community
Announcing the spring 2025 cohort of Vercel's Open Source Program. Open source community frameworks, libraries, and tools we rely on every day to build the web,

### [Introducing the Flags Explorer, first-party integrations, and updates to the Flags SDK ](/blog/introducing-the-flags-explorer-first-party-integrations-and-updates)
Published: 2025-05-07
Category: Company News
Introducing first-party integrations, the Flags Explorer, and improvements to the Flags SDK to improve feature flag workflow on Vercel.

### [Join the Vercel AI Accelerator](/blog/join-the-vercel-ai-accelerator)
Published: 2025-05-06
Category: Company News
A six-week program to help you scale your AI company offering over $4M in credits from Vercel, v0, AWS, and leading AI platforms

### [How v0 is building SEO-optimized sites by default](/blog/how-v0-is-building-seo-optimized-sites-by-default)
Published: 2025-05-02
Category: v0
Understanding how v0 ensures everything you create is seo-ready by default, without changing how you build

### [iOS developers can now offer commission-free payments on web](/blog/ios-developers-can-now-offer-commission-free-payments-on-web)
Published: 2025-05-01
The open web wins: A U.S. court ended Apple’s 27% fee on external payments, letting developers link freely and offer better, direct checkout experiences.

### [Bot Protection: One-click managed ruleset now in public beta](/blog/one-click-bot-protection-now-in-public-beta)
Published: 2025-04-23
Category: Security
Mitigate unwanted bot traffic by challenging requests from non-browser sources. Now available in public beta and free for all users on all plans.

### [Becoming an AI engineering company](/blog/becoming-an-ai-engineering-company)
Published: 2025-04-18
Category: Field Engineering
The question isn’t if AI will transform your business, but how to use it to maintain your competitive edge. Vercel CTO Malte Ubl shows how companies can understand AI’s fundamental shifts, adapt their skills, and integrate AI into their businesses.

### [Life of a Vercel request: Application-aware routing](/blog/life-of-a-request-application-aware-routing)
Published: 2025-04-15
Vercel's gateway leverages framework-defined infrastructure to intelligently load balance, protect, and route applications at-scale and with any architecture complexity.

### [Update on Spain and LALIGA blocks of the internet](/blog/update-on-spain-and-laliga-blocks-of-the-internet)
Published: 2025-04-15
A Spanish court has empowered LALIGA to block entire IP addresses tied to unauthorized football streams—causing legitimate websites hosted on Vercel to become inaccessible in Spain.

### [Migrating Grep from Create React App to Next.js](/blog/migrating-grep-from-create-react-app-to-next-js)
Published: 2025-04-14
Category: Field Engineering
We migrated grep​.app from Create React App to Next.js. We break down how we combined single-page app speed with React Server Component efficiency. 70% faster First Contentful Paint, 73% quicker network request completion, now searching 1M repos.

### [Introducing Chatbot Template](/blog/introducing-chatbot)
Published: 2025-04-09
Category: Field Engineering
Chatbot is a free, open-source template, powered by Next.js and the AI SDK, that helps you build chatbot applications.

### [Expanding observability on Vercel](/blog/expanding-observability-on-vercel)
Published: 2025-04-08
Category: Company News
The Vercel Marketplace adds new native integrations from Sentry, Checkly, and Dash0. Use the tools you already trust to monitor, measure, and debug your apps with integrated billing, single sign-on, and access to provider dashboards from Vercel.

### [Protectd: Evolving Vercel’s always-on denial-of-service mitigations](/blog/protectd-evolving-vercels-always-on-denial-of-service-mitigations)
Published: 2025-04-07
Category: Security
Protectd is our new real-time security engine that blocks DDoS attacks faster than ever—built to detect, learn from, and stop threats before they reach your app. Now powering sub-second protection across all regions by default.

### [How PAIGE grew revenue by 22% with Shopify, Next.js, and Vercel](/customers/how-paige-grew-revenue-by-22-with-shopify-next-js-and-vercel)
Published: 2025-04-03
Category: Customer stories
 Seeking an improved online experience, PAIGE reimagined their ecommerce strategy by simplifying their headless tech stack—one powered by Shopify, Next.js, and Vercel—that ultimately boosted their revenue by 22% and increased conversion rates by 76%.

### [The no-nonsense guide to composable commerce](/blog/the-no-nonsense-guide-to-composable-commerce)
Published: 2025-04-01
Composable commerce projects frequently become overly complex, leading to missed objectives and unnecessary costs. At Vercel, we take a no-nonsense approach to composable commerce that's solely focused on business outcomes. 

### [Postmortem on Next.js Middleware bypass](/blog/postmortem-on-next-js-middleware-bypass)
Published: 2025-03-25
Category: Security
Last week, we published CVE-2025-29927 and patched a critical severity vulnerability in Next.js. Here’s our post-incident analysis and next steps.

### [AI SDK 4.2](/blog/ai-sdk-4-2)
Published: 2025-03-21
Category: Field Engineering
AI SDK 4.2 introduces MCP clients, reasoning, image generation with language models, message parts, sources, and more

### [xAI and Vercel partner to bring zero-friction AI to developers](/blog/xai-and-vercel-partner-to-bring-zero-friction-ai-to-developers)
Published: 2025-03-20
Category: Company News
Vercel partners with xAI to bring Grok models directly to your Vercel projects through the Vercel Marketplace—and soon v0—with no additional signup required. xAI adds a new free tier through Vercel to enable quick prototyping and experimentation.

### [Jeanne DeWitt Grosser joins Vercel as COO](/blog/jeanne-dewitt-grosser-joins-vercel-as-coo)
Published: 2025-03-13
Category: Company News
Vercel welcomes Jeanne DeWitt Grosser as Chief Operating Officer. Jeanne helped take Stripe from $100M to billions, pioneered usage-based go-to-market strategies, and scaled influential developer platforms. Now, she brings that momentum to Vercel.

### [Personalization strategies that power ecommerce growth](/blog/personalization-strategies-that-power-ecommerce-growth)
Published: 2025-03-07
Category: Field Engineering
Learn how to implement high-performance personalization with Next.js and Vercel. Discover best practices, avoid common pitfalls, and deliver fast, scalable, and revenue-driving ecommerce experiences without sacrificing speed or user experience.

### [How Fluid compute works on Vercel](/blog/how-fluid-compute-works-on-vercel)
Published: 2025-03-03
Category: Field Engineering
See how Fluid combines server efficiency and serverless flexibility by reusing compute before creating new resources, reducing cold starts, and running in unconstrained environments while staying secure and fast. Cut compute costs by up to 85%.

### [Using the AI SDK to build Sitecore Stream's AI-powered brand aware assistant](/customers/using-the-ai-sdk-to-build-sitecore-streams-ai-powered-brand-aware-assistant)
Published: 2025-03-03
Category: Customer stories
Sitecore Stream from Sitecore, powered by the AI SDK, empowers marketers with real-time conversational AI. 

### [Integrating Vercel and Sitecore for 2x faster development times and 111% higher conversions](/customers/integrating-vercel-and-sitecore-for-2x-faster-development-times-and-111)
Published: 2025-02-24
Category: Customer stories
Learn how Avanade achieved 90+ Lighthouse Scores by moving away from monolith legacy systems to Sitecore XM Cloud and Vercel.

### [Vercel security roundup: Faster defenses and better visibility for your apps](/blog/vercel-security-roundup-faster-defenses-and-better-visibility-for-your-apps)
Published: 2025-02-21
Category: Security
Learn how Vercel's security product updates improved traffic visibility, enhanced mitigation techniques, and resulted in blocking billions of malicious attacks.

### [Bridging the gap between design and code with v0](/customers/bridging-the-gap-between-design-and-code-with-v0)
Published: 2025-02-12
Category: Customer stories
Understanding how the team at Speakeasy uses v0 to ship faster with features like Figma import and custom Tailwind config. 

### [Introducing Fluid compute](/blog/introducing-fluid-compute)
Published: 2025-02-04
Category: Company News
Fluid compute on Vercel combines serverless efficiency with server-like flexibility, reducing cold starts and cutting compute costs by up to 85%. Scale intelligently, minimize latency, and optimize performance with zero config.

### [ISR on Vercel is now faster and more cost-efficient](/blog/isr-on-vercel-is-now-faster-and-more-cost-efficient)
Published: 2025-01-30
Category: Field Engineering
We've optimized how ISR cache updates are managed on Vercel, making them faster and more cost-efficient. 

### [Working with Figma and custom design systems in v0](/blog/working-with-figma-and-custom-design-systems-in-v0)
Published: 2025-01-27
Category: v0
Learn best practices on importing your designs from Figma, working with shadcn/ui, and leveraging public npm packages. 

### [Mitigating Denial of Wallet risks with Vercel](/blog/mitigating-denial-of-wallet-risks-with-vercel)
Published: 2025-01-24
Category: Security
Protect against Denial of Wallet (DoW) attacks with Vercel. DoW exploits cloud scalability to inflate costs. Vercel provides solutions like budget alerts, spend limits, and anomaly detection to safeguard workloads and maintain financial stability.

### [Vercel acquires Tremor to invest in open source React components](/blog/vercel-acquires-tremor)
Published: 2025-01-22
Category: Company News
Tremor, a library of React components to build charts and dashboards, joins Vercel. With this acquisition, all Tremor products—including Tremor Blocks—are now free and open source, bringing elegant UI components to all developers.

### [AI SDK 4.1](/blog/ai-sdk-4-1)
Published: 2025-01-20
Category: Field Engineering
AI SDK 4.1 introduces image generation, non-blocking data streaming, improved tool calling, and more

### [Transforming how you work with v0](/blog/transforming-how-you-work-with-v0)
Published: 2025-01-10
Category: v0
v0 lets all creators—not just developers—bring their ideas to life. Explore v0 use cases and prompt inspiration for designers, marketers, project managers, customer support, data analysis, and more.

### [Headless Salesforce: An incremental migration from monolith to composable](/blog/salesforce-incremental-migration)
Published: 2025-01-07
Go headless with your Salesforce storefront while keeping your commerce backend intact. Learn how to de-risk migration to Next.js and Vercel, reducing load times & boosting conversions.

### [Building the Black Friday-Cyber Monday live dashboard](/blog/building-the-black-friday-cyber-monday-live-dashboard)
Published: 2024-12-24
See how we built the data-heavy Black Friday-Cyber Monday dashboard to be cost-efficient, fast, and accurate. Building a data-heavy, real-time dashboard with a good user experience comes with challenges. Let's walk through how we overcame them.

### [Optimizing secure build infrastructure with Secure Compute](/blog/optimizing-secure-builds-with-hive-and-secure-compute)
Published: 2024-12-18
Category: Field Engineering
We built Hive, our general compute platform, after outgrowing off-the-shelf solutions.  See how it's now powering secure connections to private networks, cutting initialization from 90s to 5s and improving build speeds by 30%.

### [The rise of the AI crawler](/blog/the-rise-of-the-ai-crawler)
Published: 2024-12-17
New research reveals how ChatGPT, Claude, and other AI crawlers process web content, including JavaScript rendering, assets, and other behavior and patterns—with recommendations for site owners, devs, and AI users.

### [Technical audits: Optimizing cost, performance, and productivity](/blog/technical-audits)
Published: 2024-12-12
See what we've learned from hundreds of real-world audits—what to look for and what you can do to improve your applications.

### [Extra Space Storage's build times became 17x faster with Vercel](/customers/extra-space-storages-build-times-became-17x-faster-with-vercel)
Published: 2024-12-11
Category: Customer stories
Extra Space Storage cut build times by 95% after migrating to Next.js on Vercel.   See how features like Incremental Static Regeneration and per-branch environments enable faster and more confident releases.

### [Vercel and AWS partner on AI tools and experiences](/blog/vercel-and-aws-partner-on-ai-tools-and-experiences)
Published: 2024-12-09
Category: Company News
Vercel has been selected for a Strategic Collaboration Agreement (SCA) partnership with AWS—to deliver the next generation of AI-enabled developer tools

### [Life of a Vercel request: Securing your app's traffic with Vercel](/blog/life-of-a-request-securing-your-apps-traffic-with-vercel)
Published: 2024-12-05
Category: Security
The Vercel Firewall automatically prevents over 1B malicious requests every week, with 5x that amount coming in over Black Friday-Cyber Monday.  Learn how Vercel protects every request so you can focus on building, not fighting attacks.

### [Billions of dollars, billions of requests: Black Friday-Cyber Monday 2024](/blog/black-friday-cyber-monday-2024-recap)
Published: 2024-12-03
With 3B+ firewall blocks and 99.999+% uptime, top ecommerce brands like Under Armour, Fanatics, and ASICS trust Vercel’s Managed Infrastructure to handle the demand and ship with confidence.

### [Retailer sees $10M increase in sales on Vercel](/customers/retailer-sees-10m-increase-in-sales-on-vercel)
Published: 2024-11-27
Category: Customer stories
A global sportswear retailer had a record-breaking Black Friday-Cyber Monday last year, which increased sales by $10M: 33% increase in average orders per minute, 500ms reduction in Time to First Byte, 2% increase in overall conversions.

### [From minutes to seconds: How Meter accelerates delivery with Vercel and Next.js](/customers/from-minutes-to-seconds-how-meter-accelerates-delivery-with-vercel-and-next)
Published: 2024-11-26
Category: Customer stories
Meter migrated from AWS to Vercel, cutting build times from 10+ minutes to under a minute while unifying their monorepo and shipping Command, an AI-powered network management tool built on Next.js.

### [How Notion powers rapid and performant experimentation](/customers/how-notion-powers-rapid-and-performant-experimentation)
Published: 2024-11-25
Category: Customer stories
Notion runs hundreds of experiments per year with Statsig and Next.js on Vercel. Learn how they maintain Core Web Vitals on marketing pages while testing new content with users.

### [Life of a Vercel request: Navigating the Edge Network](/blog/life-of-a-vercel-request-navigating-the-edge-network)
Published: 2024-11-21
Unpacking the core usage metrics of Edge Requests and Fast Data Transfer. Learn how Vercel handles network routing and data transfer—while giving you full control over performance and costs.

### [Vercel acquires Grep to accelerate code search](/blog/vercel-acquires-grep)
Published: 2024-11-20
Category: Company News
Announcing the acquisition of Grep to further our mission of helping developers work and ship faster. 

### [AI SDK 4.0](/blog/ai-sdk-4-0)
Published: 2024-11-18
Category: Field Engineering
Introducing PDF support, computer use, and an xAI Grok provider

### [Accelerating partner success: Vercel’s new Partner Program benefits](/blog/vercel-partner-program-updates)
Published: 2024-11-15
At Vercel, we see partnership and collaboration as keys to driving innovation and customer success. Vercel partners with consultants, agencies, global system integrators, tech partners, and cloud hyperscalers.

### [Life of a Vercel request: What happens when a user presses enter](/blog/life-of-a-vercel-request-what-happens-when-a-user-presses-enter)
Published: 2024-11-13
Application delivery through the lens of a web request with Vercel’s framework-defined infrastructure.

### [Vercel named a Visionary in 2024 Gartner® Magic Quadrant™ for Cloud Application Platforms](/blog/vercel-named-a-visionary-in-2024-gartner-magic-quadrant-for-cloud)
Published: 2024-11-08
Category: Company News
Vercel is proud to be recognized as a Visionary; we view this Gartner® Magic Quadrant™ as a validation of the Frontend Cloud ecosystem.

### [MotorTrend: Shifting into overdrive with Vercel](/customers/motortrend-shifting-into-overdrive-with-vercel)
Published: 2024-11-07
Category: Customer stories
How a performance-first approach drives business value. By migrating to Vercel, MotorTrend improved release times from 18 days to 10 minutes. As velocity improved, ad impressions grew, translating into real business value.

### [Break the news, not the site: Leading news organizations upgrade their infrastructure ahead of the election](/customers/break-the-news-not-the-site)
Published: 2024-10-31
Category: Customer stories
Code and Theory partners with Vercel to deliver breaking news with high performance and uptime. News organizations RealClearPolitics and Minnesota Star Tribune upgrade their digital infrastructure to better serve their audience during traffic surges.

### [A deep dive into Vercel’s build infrastructure](/blog/a-deep-dive-into-hive-vercels-builds-infrastructure)
Published: 2024-10-30
Category: Field Engineering
Vercel’s low-level untrusted and ephemeral compute platform is designed to give us the control needed to securely and efficiently manage and run builds.

### [Recap: Next.js Conf 2024](/blog/recap-next-js-conf-2024)
Published: 2024-10-25
Category: Community
Next.js Conf 2024

### [What's new in Svelte 5](/blog/whats-new-in-svelte-5)
Published: 2024-10-23
Svelte 5 brings runes for universal reactivity, snippets for reusable markup, and compiler improvements. Get started with Svelte 5 on Vercel today.

### [Maximizing outputs with v0: From UI generation to code creation](/blog/maximizing-outputs-with-v0-from-ui-generation-to-code-creation)
Published: 2024-10-23
Category: Field Engineering
Learn prompt engineering best practices for working with v0's core functionality to get the best results.

### [BNP Paribas Open: Serving up scores and experiences in real time with Work & Co and Vercel](/customers/bnp-paribas-open-serving-up-scores-and-experiences-in-real-time-with-work)
Published: 2024-10-22
Category: Customer stories
Work & Co partners with Vercel for BNP Paribas Open's digital transformation resulting in better performance and velocity. 

### [How Vercel adopted microfrontends](/blog/how-vercel-adopted-microfrontends)
Published: 2024-10-22
Category: Field Engineering
Learn how Vercel cut build times and improved developer velocity while maintaining a smooth user experience with microfrontends.

### [Eval-driven development: Build better AI faster](/blog/eval-driven-development-build-better-ai-faster)
Published: 2024-10-17
Category: Field Engineering
Learn how eval-driven development helps you build better AI faster. Discover a new testing paradigm for AI-native development and unlock continuous improvement.

### [v0 plans for teams are here](/blog/v0-plans-for-teams)
Published: 2024-10-15
Category: v0
Introducing v0 Team and Enterprise plans—designed for secure and efficient collaboration. Team can share Projects, chats, and resources with higher messaging limits. Enterprises can secure access with SSO and automatic opt-out of data training.

### [Add 3D to your web projects with v0 and React Three Fiber](/blog/add-3d-to-your-web-projects-with-v0-and-react-three-fiber)
Published: 2024-10-10
Category: Field Engineering
How to build 3D web projects with React Three Fiber in v0

### [How Emburse increased site performance by 4x with Vercel](/customers/how-emburse-increased-site-performance-by-4x-with-vercel)
Published: 2024-10-10
Category: Customer stories
Emburse transformed its digital presence by adopting Sanity, Next.js, and Vercel, enhancing performance, SEO, and global localization. This allowed their marketing team to manage content and launch campaigns more efficiently.

### [Leveraging Vercel and the AI SDK to deliver a seamless, AI-powered experience as a solo founder](/customers/leveraging-vercel-and-the-ai-sdk-to-deliver-a-seamless-ai-powered-experience)
Published: 2024-10-09
Category: Customer stories
How ChatPRD Scaled to 20,000 users with Vercel and the AI SDK. ChatPRD is an AI co-pilot designed for product managers, enabling them to write product requirements documents, brainstorm roadmaps, and improve overall efficiency around product work.

### [How Chatbase scaled rapidly with Vercel's developer experience and AI SDK](/customers/how-chatbase-scaled-rapidly-with-vercels-developer-experience-and-ai-sdk)
Published: 2024-10-09
Category: Customer stories
Scaling rapidly in the AI market with Vercel, Next.js, and the AI SDK 

### [How Supabase increased signups through the Vercel Marketplace](/blog/how-supabase-increased-signups-through-the-vercel-marketplace)
Published: 2024-10-07
Supabase increased signups through the Vercel Marketplace

### [Navigating Web3 dynamism: Ledger's solution to traffic spike stability with Vercel](/customers/ledgers-solution-to-traffic-spike-stability-with-vercel)
Published: 2024-10-04
Category: Customer stories
Discover how Ledger, a leader in hardware wallets, transformed their online presence to handle unpredictable Web3 traffic spikes. Learn how Vercel and Next.js improved their performance, boosting release frequency by 200x and reducing load times by 67%.

### [Serverless servers: Efficient serverless Node.js with in-function concurrency](/blog/serverless-servers-node-js-with-in-function-concurrency)
Published: 2024-10-03
Building a compute layer that is highly-optimized for interactive workloads, server-rendering, and APIs

### [Vercel WAF upgrade brings persistent actions, rate limiting, and API control](/blog/vercel-waf-upgrade-brings-persistent-actions-rate-limiting-and-api-control)
Published: 2024-10-02
New capabilities reduce effects of DDoS attacks and enhance traffic control

### [Accelerating developer velocity and creating high-impact web teams](/blog/accelerating-developer-velocity-and-creating-high-impact-web-teams)
Published: 2024-09-27
Transforming team dynamics by shifting focus from infrastructure management to innovation and delivering value.

### [Preventing infrastructure abuse with Vercel Firewall](/blog/preventing-infrastructure-abuse-with-vercel-firewall)
Published: 2024-09-24
Category: Security
DDoS resilience in the face of modern threats on the Vercel Firewall to automatically shield customers and maintain service availability

### [AI SDK 3.4](/blog/ai-sdk-3-4)
Published: 2024-09-20
Category: Field Engineering
AI SDK 3.4 introduces middleware, data stream protocol, and multi-step generations

### [From CDNs to Frontend Clouds](/blog/from-cdns-to-frontend-clouds)
Published: 2024-09-20
Content Delivery Networks (CDNs) and Infrastructure as code (IaC) evolved into the next generation of web application delivery.

### [Managing 275 thousand pages and 8 million assets at top speed with ISR](/customers/managing-275-thousand-pages-and-8-million-assets-with-isr)
Published: 2024-09-17
Category: Customer stories
Content-heavy websites don't need to suffer from long build times. With Incremental Static Regeneration (ISR), Mecum Auction Company handles 8M digital assets and 275K pages, while improving performance across 120M page views.

### [ISR: A flexible way to cache dynamic content](/blog/isr-a-flexible-way-to-cache-dynamic-content)
Published: 2024-09-16
Explore how Incremental Static Regeneration (ISR) enhances your caching strategy. Learn its benefits, implementation across frameworks, and real-world applications. Optimize performance with this hybrid approach to dynamic content delivery.

### [Deploying dreams: An inside look at a summer internship with Vercel](/blog/summer-internship-at-vercel)
Published: 2024-09-13
Category: Field Engineering
What's an internship like at Vercel? Hear firsthand from one of Vercel's summer interns what the process was like, what they worked on, and what they learned.

### [What’s new in React 19](/blog/whats-new-in-react-19)
Published: 2024-09-04
React 19 is near. Here's what to expect and how you can get started deploying React 19 on Vercel.

### [Transforming customer support with AI: How Vercel decreased tickets by 31%](/blog/transforming-customer-support-with-ai-how-vercel-decreased-tickets)
Published: 2024-09-03
Category: Community
At Vercel, we integrated AI into our support workflow. Our AI agent reduced human-handled tickets by 31%, allowing us to maintain high support standards while serving a growing customer base.

### [Enhancing security of backend connectivity with OpenID Connect](/blog/enhancing-security-of-backend-connectivity-with-openid-connect)
Published: 2024-08-28
Vercel OpenID Connect support helps you replace long-lived credentials with temporary tokens to reduce risk.

### [Introducing the Vercel Marketplace](/blog/introducing-the-vercel-marketplace)
Published: 2024-08-28
Category: Company News
The Vercel Marketplace adds support for EdgeDB, Redis, and Supabase with unified billing and simpler installations.

### [Devolver ships game websites 73% faster with Vercel](/customers/devolver-ships-game-websites-73-faster-with-vercel)
Published: 2024-08-21
Category: Customer stories
How Devolver ships game websites 73% faster with built-in CI/CD, zero-configuration integrations, Preview URLs on each pull request, and more.

### [Using the AI SDK to fix edge-case errors in our code](/blog/using-the-ai-sdk-to-fix-edge-case-errors-in-our-code)
Published: 2024-08-15
Category: Field Engineering
Leveraging the AI SDK to build with and solve problems with AI

### [How to build scalable AI applications](/blog/how-to-build-scalable-ai-applications)
Published: 2024-08-12
Explore AI integration strategies with Vercel's AI SDK. Learn to choose providers, optimize performance, and future-proof your apps. Discover tools for seamless AI deployment and scalability.

### [Update regarding Vercel service disruption on August 7, 2024](/blog/update-regarding-vercel-service-disruption-on-august-7-2024)
Published: 2024-08-09
Category: Field Engineering
Understanding the service disruption and Vercel's next steps

### [Vercel AI SDK 3.3](/blog/vercel-ai-sdk-3-3)
Published: 2024-08-06
Category: Field Engineering
Vercel AI SDK 3.3 introduces tracing, multi-modal attachments, JSON streaming to clients, and more.

### [How to integrate AI into your business](/blog/how-to-integrate-ai-into-your-business)
Published: 2024-08-06
Learn to build robust AI use cases, evaluate initiatives, and integrate AI into your team's workflow. Discover how Vercel's platform accelerates AI development and drives innovation.

### [Protecting your app (and wallet) against malicious traffic](/blog/protecting-your-app-and-wallet-against-malicious-traffic)
Published: 2024-08-02
Category: Security
Learn how to block traffic with the Firewall, set up soft and hard spend limits, apply code-level optimizations, and more.

### [Achieving feature rollouts with ultra-low latency and zero impact to conversion](/customers/beyond-menu-scaling-with-hypertune-and-vercel)
Published: 2024-08-01
Category: Customer stories
 Learn how Beyond Menu resolved feature flagging and A/B testing issues in serverless environments by integrating Hypertune with Vercel’s Edge Config, achieving seamless performance and improved user experience.

### [How Google handles JavaScript throughout the indexing process](/blog/how-google-handles-javascript-throughout-the-indexing-process)
Published: 2024-07-31
Category: Field Engineering
Over the years, Google's treatment of JavaScript has changed, leaving us with misconceptions of how it's indexed. Here, we debunk the myths.

### [Flags as code in Next.js](/blog/flags-as-code-in-next-js)
Published: 2024-07-26
Category: Field Engineering
The Flags SDK is a free open-source library that gives developers the tools they need to use feature flags in Next.js and SvelteKit applications.

### [Elkjøp's Digital Transformation: Powering Retail Innovation with Next.js and Vercel](/customers/elkjops-digital-transformation-with-next-js-and-vercel)
Published: 2024-07-24
Category: Customer stories
By adopting Next.js and Vercel, Elkjøp improved page loads, SEO, and overall user experience, driving over $1 Billion in digital revenue

### [Turbopack updates: Moving homes](/blog/turbopack-moving-homes)
Published: 2024-07-23
Category: Field Engineering
An update on our progress with Turbopack, details on where we’re headed, and a planned move of Turbopack to the Next.js repository.

### [How to choose the best rendering strategy for your app](/blog/how-to-choose-the-best-rendering-strategy-for-your-app)
Published: 2024-07-23
Demystify Next.js rendering strategies: SSG, SSR, CSR, ISR, and PPR. Optimize your web apps for performance, SEO, and user experience. Learn when and how to use each approach with real-world examples and practical tips for modern web development.

### [Understanding Vercel Functions](/blog/understanding-vercel-functions)
Published: 2024-07-05
Category: Field Engineering
Learn about how Vercel Functions help you run secure, highly available, and fast compute.

### [Function streaming to be framework-agnostic on Vercel](/blog/vercel-functions-streaming-to-be-framework-agnostic)
Published: 2024-07-04
Category: Field Engineering
Function streaming now framework-agnostic on Vercel.

### [Introducing Vercel AI SDK 3.2](/blog/introducing-vercel-ai-sdk-3-2)
Published: 2024-06-18
Category: Field Engineering
Vercel AI SDK 3.2 enables agent and embeddings workflows while improving provider support and DX. 

### [Getting started with AI: Advice from the experts at Vercel Ship](/blog/getting-started-with-ai-advice-from-the-experts-at-vercel-ship)
Published: 2024-06-13
Takeaways from the AI Enterprise Panel at Vercel Ship 2024.

### [Demystifying INP: New tools and actionable insights](/blog/demystifying-inp-new-tools-and-actionable-insights)
Published: 2024-06-12
Category: Field Engineering
Deep dive into Interaction to Next Paint (INP) optimization: A technical guide exploring real-world strategies used to improve INP on nextjs.org. Learn how to tackle common challenges and enhance responsiveness in your web applications.

### [Never drop the illusion: How Frame.io builds fluid user experiences](/customers/frameio-never-drop-the-illusion)
Published: 2024-06-11
Category: Customer stories
Their users "see in milliseconds," so every frame within their web experience matters. Frame.io commits itself to delivering web applications that feel as responsive and powerful as their desktop counterparts.

### [Mintlify: Scaling a powerful documentation platform with Vercel](/customers/mintlify-scaling-a-powerful-documentation-platform-with-vercel)
Published: 2024-06-03
Category: Customer stories
Learn how Mintlify built and scaled their multi-tenant docs platform with custom domains

### [Introducing bytecode caching for Vercel Functions](/blog/introducing-bytecode-caching-for-vercel-functions)
Published: 2024-06-03
Category: Field Engineering
Vercel's bytecode caching eliminates JavaScript compilation on cold starts by caching V8 bytecode across invocations, reducing TTFB by up to 27% and billed duration by up to 58% for serverless functions.

### [Vercel Ship 2024 recap](/blog/vercel-ship-2024)
Published: 2024-05-24
Category: Company News
Vercel Ship 2024 highlighted the integrations, ecosystem, and teams building the web's best products. Read the recap to catch up on all our announcements.

### [Introducing the Vercel Web Application Firewall](/blog/introducing-the-vercel-waf)
Published: 2024-05-23
Category: Company News
Introducing the Vercel Web Application Firewall: application-aware, Vercel-native protection that brings the web one step closer to being secure by default.

### [Shipping safer and smarter: Integrating feature flags deeper in the Vercel workflow](/blog/feature-flags)
Published: 2024-05-23
Category: Company News
Introducing a platform-wide understanding of feature flags in Vercel, and an experimental Next.js design pattern for working with flags in code.

### [Introducing new developer tools in the Vercel Toolbar](/blog/introducing-new-developer-tools-in-the-vercel-toolbar)
Published: 2024-05-23
Category: Company News
The Vercel Toolbar now includes even more developer tools to simplify collaboration, improve accessibility, and enhance your workflow.

### [Securing data in your Next.js app with Okta and OpenFGA](/blog/securing-data-in-your-next-js-app-with-okta-and-openfga)
Published: 2024-05-16
Category: Community
Learn how to integrate a Data Access Layer in your Next.js app and use it to implement a fine-grained authorization model with OpenFGA.

### [How Vercel helped Desenio future-proof their business](/customers/how-vercel-helped-desenio-future-proof-their-business)
Published: 2024-05-09
Category: Customer stories
Vercel helped Desenio future-proof their business

### [7 AI features you can add to your app today](/blog/7-ai-features-you-can-add-to-your-app-today)
Published: 2024-05-09
Discover how AI is transforming businesses with Vercel. Learn about easy LLM integration, 7 game-changing applications, and how to implement AI features to boost user experience and growth.

### [Vercel Functions are now faster—and powered by Rust](/blog/vercel-functions-are-now-faster-and-powered-by-rust)
Published: 2024-05-03
Category: Field Engineering
Learn about how we've improved startup performance with our Rust-powered functions.

### [How Dub grew to 3,000 active domains with Vercel’s multi-tenant SaaS toolkit ](/customers/how-dub-grew-to-3000-active-domains-with-vercels-multi-tenant-saas-toolkit)
Published: 2024-05-03
Category: Customer stories
The open-source link management platform Dub boasts over 3,000 active domains—and growing—with Vercel's multi-tenant SaaS toolkit. Learn how.

### [Vercel AI SDK 3.1: ModelFusion joins the team](/blog/vercel-ai-sdk-3-1-modelfusion-joins-the-team)
Published: 2024-05-02
Category: Field Engineering
ModelFusion joins Vercel 

### [Vercel supports HIPAA compliance](/blog/vercel-supports-hipaa-compliance)
Published: 2024-05-01
Category: Security
Vercel now supports HIPAA compliance for our enterprise customers, enabling companies to leverage our Frontend Cloud while maintaining compliance

### [How Vercel helped Tonies expand into new markets and improve conversion rates](/customers/how-vercel-helped-tonies-expand-into-new-markets)
Published: 2024-04-26
Category: Customer stories
By embracing a headless stack powered by Vercel and Contentful, Tonies was able to scale their ecommerce presence, expand into new markets, and improve UX.

### [Latency numbers every frontend developer should know](/blog/latency-numbers-every-web-developer-should-know)
Published: 2024-04-23
Category: Field Engineering
Latency numbers every web developer should know

### [How Global Retail Brands cut development time from months to 1 week with Vercel](/customers/how-global-retail-brands-cut-development-time-from-months-to-1-week)
Published: 2024-04-18
Category: Customer stories
Learn how GRB, one of Australia's fastest-growing retailers, improved site uptime, development time, and overall developer experience with Vercel.

### [Building an interactive 3D event badge with React Three Fiber](/blog/building-an-interactive-3d-event-badge-with-react-three-fiber)
Published: 2024-04-17
Category: Field Engineering
See a full working demo of how we built the interactive Vercel Ship '24 badge using React Three Fiber and react-three-rapier.

### [Releasing safe and cost-efficient blue-green deployments](/blog/releasing-safe-and-cost-efficient-blue-green-deployments)
Published: 2024-04-12
Category: Field Engineering
Learn how Vercel's platform primitives enable safe and scalable blue-green deployments that mitigate the risks of rolling out new software versions.

### [Creating a robust platform for documentation with Next.js and Vercel](/customers/creating-a-robust-platform-for-documentation-with-next-js-and-vercel)
Published: 2024-04-10
Category: Customer stories
Learn how Teleport unlocked a new customer acquisition channel by overhauling and migrating their documentation to Next.js and Vercel. 

### [Composable AI for ecommerce: Hands-on with Vercel’s AI SDK](/blog/composable-ai-for-ecommerce-hands-on-with-vercels-ai-sdk)
Published: 2024-04-09
Category: Field Engineering
With v0 and the Vercel AI SDK, you can go from AI pipe dream to working prototype in just a few hours. See how AI can transform your ecommerce storefront.

### [How Ruggable saw 300% more organic clicks by optimizing their frontend architecture](/customers/how-ruggable-saw-more-organic-clicks-by-optimizing-their-frontend)
Published: 2024-04-08
Category: Customer stories
Ruggable improved organic search traffic by 300% by migrating their ecommerce storefront from a Shopify monolith to a headless solution on Next.js and Vercel.

### [Improved infrastructure pricing](/blog/improved-infrastructure-pricing)
Published: 2024-04-04
Category: Company News
We're reducing pricing on Vercel fundamentals like bandwidth and functions.

### [Design Engineering at Vercel](/blog/design-engineering-at-vercel)
Published: 2024-03-29
Category: Field Engineering
Design Engineers at Vercel blend aesthetic sensibility with technical skills. Learn about Vercel's philosophy on what Design Engineering is and how we work.

### [Demant achieves global scalability and 30x faster response times with Vercel](/customers/demant-achieves-global-scalability-and-30x-faster-response-times-with-vercel)
Published: 2024-03-29
Category: Customer stories
Global scalability and 30x faster response times with Vercel

### [Protecting AI apps from bots and bad actors with Vercel and Kasada](/blog/protecting-ai-apps-with-vercel-and-kasada)
Published: 2024-03-22
Category: Field Engineering
Learn how Vercel protects the AI SDK Playground using our best-in-class DDoS mitigation, Next.js Middleware, and our partner Kasada.

### [Revolutionizing video editing on the web with Next.js and Vercel](/customers/revolutionizing-video-editing-on-the-web-with-next-js-and-vercel)
Published: 2024-03-20
Category: Customer stories
Ozone, the AI-infused, web-based video editor, accelerated their development process by 5x after switching to Vercel and Next.js. See how.

### [Leonardo generates 4.5M images daily with Next.js and Vercel](/customers/leonardo-ai-performantly-generates-4-5-million-images-daily-with-next-js-and-vercel)
Published: 2024-03-18
Category: Customer stories
Learn how Leonardo.Ai leveraged Vercel to reduce build times, speed up page performance, and improve their developer experience.

### [WordPress monolith to Vercel: How Personio elevated site performance and efficiency](/customers/from-wordpress-monolith-to-vercel-personio-elevates-site-performance)
Published: 2024-03-18
Category: Customer stories
Personio migrated from a WordPress monolith to a composable solution with Vercel and Next.js, improving performance, security, and iteration speed. 

### [8 advantages of composable commerce](/blog/8-advantages-of-composable-commerce)
Published: 2024-03-07
Faster development cycles, better personalization, increased site performance, and more. Here are the benefits composable commerce can offer.

### [Introducing feature flag management from the Vercel Toolbar](/blog/toolbar-feature-flags)
Published: 2024-03-06
Category: Company News
Uplevel your flags workflow with the Vercel Toolbar

### [Introducing AI SDK 3.0 with Generative UI support](/blog/ai-sdk-3-generative-ui)
Published: 2024-03-01
Category: Field Engineering
Stream React Components from LLMs to deliver richer user experiences

### [The Frontend Cloud: Powering resiliency for global web applications](/blog/the-resiliency-of-the-frontend-cloud)
Published: 2024-02-29
Category: Field Engineering
Optimize your web presence for maximum uptime, scalability, and security. Discover the power of frontend clouds for enterprise resilience.

### [Deploying safely on Vercel without merge queues](/blog/deploy-safely-on-vercel-without-merge-queues)
Published: 2024-02-26
Category: Field Engineering
Deploy quickly and safely with Vercel without merge queues

### [Effortless high availability for dynamic frontends](/blog/effortless-high-availability-for-dynamic-frontends)
Published: 2024-02-21
Category: Field Engineering
Vercel's Frontend Cloud is designed with high availability at its core, to maintain uptime for dynamic applications.

### [Evolving Vercel Functions](/blog/evolving-vercel-functions)
Published: 2024-02-14
Category: Field Engineering
Our first major iteration of Vercel Functions with increased concurrency, longer durations, faster cold starts, streaming, and more.

### [Vercel + WPP: World-class creativity enabled by technology](/blog/vercel-wpp-creativity-enabled-by-technology)
Published: 2024-02-14
Category: Company News
Vercel and WPP: World-class creativity enabled by technology, with six trends driving the future of the web.

### [Finishing Turborepo's migration from Go to Rust](/blog/finishing-turborepos-migration-from-go-to-rust)
Published: 2024-02-12
Category: Field Engineering
We've finished porting Turborepo, the high performance JavaScript and TypeScript build system, from Go to Rust.

### [Introducing AI Integrations on Vercel](/blog/ai-integrations)
Published: 2024-02-08
Category: Company News
Vercel's AI Integrations and AI SDK connect your frontend to leading AI providers for text, image, and audio models, making it easy to build streaming chatbots, RAG systems, and generative experiences in Next.js.

### [PCI compliance for ecommerce](/blog/pci-compliance-for-ecommerce-teams)
Published: 2024-02-07
Category: Security
Leverage iframes for payment processing to enable PCI compliance and maintain secure transactions on Vercel.

### [How streaming helps build faster web applications](/blog/how-streaming-helps-build-faster-web-applications)
Published: 2024-01-31
Category: Field Engineering
Learn why streaming is critical for performance and how the Next.js App Router integrates with React Suspense to easily enable streaming UI.

### [How Core Web Vitals affect SEO](/blog/how-core-web-vitals-affect-seo)
Published: 2024-01-19
Category: Field Engineering
Understand your application's Google page experience ranking and Lighthouse scores. We'll dive into what they are, how they’re measured, and how your users and search ranking are impacted by them.

### [Architecting a live look at reliability: Stripe's viral Black Friday site](/customers/architecting-reliability-stripes-black-friday-site)
Published: 2024-01-16
Category: Customer stories
Stripe built a real-time Black Friday microsite on Vercel in 19 days, using Next.js with ISR and SWR to handle peak loads of 93,304 transactions per minute while maintaining 99.999% API uptime.

### [Common mistakes with the Next.js App Router and how to fix them](/blog/common-mistakes-with-the-next-js-app-router-and-how-to-fix-them)
Published: 2024-01-08
Category: Field Engineering
Learn how to use the Next.js App Router more effectively and understand the new model.

### [Forrester Total Economic Impact™ study: Vercel delivered a 264% ROI](/blog/forrester-total-economic-impact-vercel-ROI)
Published: 2024-01-04
Category: Company News
Forrester's Total Economic Impact study found Vercel's Frontend Cloud delivers a three-year 264% ROI and $9.53M in benefits, with 90% less infrastructure management time and 4x more major releases.

### [The developer experience of the Frontend Cloud](/blog/the-developer-experience-of-the-frontend-cloud)
Published: 2023-12-21
Category: Field Engineering
Vercel's Frontend Cloud offers a complete Developer Experience (DX) Platform, to automate complex systems and offer best-in-market iteration velocity.

### [AWS re:Invent 2023: Iteration velocity is the solution to all software problems](/blog/aws-reinvent-2023-iteration-velocity)
Published: 2023-12-20
Category: Company News
Go composable with Vercel's Frontend Cloud to unlock faster iteration velocity, from AI-ready architecture and instant previews to generative UI with v0 that turns text prompts into production code.

### [Introducing Conformance and Code Owners: Move fast, don't break things](/blog/introducing-conformance)
Published: 2023-12-05
Category: Company News
Vercel's Conformance and Code Owners bring automated static analysis and framework-defined code ownership to   enterprise teams, catching performance and security issues before production while ensuring the right reviewers approve every change.

### [The user experience of the Frontend Cloud](/blog/the-user-experience-of-the-frontend-cloud)
Published: 2023-12-04
Category: Field Engineering
The user experience of the Frontend Cloud: Part 2 of the developer's guide to a future-proofed stack.

### [Guide to fast websites with Next.js: Tips for maximizing server speeds and minimizing client burden](/blog/guide-to-fast-websites-with-next-js-tips-for-maximizing-server-speeds)
Published: 2023-11-29
Category: Community
A collection of tips to make faster Next.js websites by maximizing work on the server and minimizing the burden on the client.

### [The power of headless: Ecommerce success with Next.js, Vercel, and Shopify](/customers/commerceui-headless-shopify-nextjs)
Published: 2023-11-28
Category: Customer stories
Commerce-UI helps designer ecommerce brands deliver a world-class experience to their online users. 

### [The foundations of the Frontend Cloud](/blog/the-foundations-of-the-frontend-cloud)
Published: 2023-11-21
Category: Field Engineering
An introduction to the underlying infrastructure of the Frontend Cloud: Part 1 of the developer's guide to a future-proofed stack for the modern frontend.

### [How to scale a large codebase](/blog/how-to-scale-a-large-codebase)
Published: 2023-11-16
Category: Field Engineering
Transition from monolithic to monorepo architectures with Vercel. Explore feature flags for safe releases, incremental builds for quick iterations, and skew protection for version consistency to ease codebase management and speed up development.

### [Partial prerendering: Building towards a new default rendering model for web applications](/blog/partial-prerendering-with-next-js-creating-a-new-default-rendering-model)
Published: 2023-11-09
Category: Field Engineering
PPR combines ultra-quick static delivery with fully dynamic capabilities and we believe it has the potential to become the default rendering model for web applications, bringing together the best of static site generation and dynamic delivery.

### [Building the most ambitious sites on the Web with Vercel and Next.js 14](/blog/building-the-most-ambitious-sites-on-the-web-with-vercel-and-next-js-14)
Published: 2023-11-06
Category: Company News
Vercel and Next.js 14: Server Actions, Experimental Partial Prerendering, Next.js faster startups, and other announcements.

### [Building secure and performant web applications on Vercel](/blog/building-secure-and-performant-web-applications-on-vercel)
Published: 2023-11-06
Vercel's Frontend Cloud offers support for deploying complex and dynamic web applications with managed infrastructure so you have control and flexibility without having to worry about configuration and maintenance

### [Understanding cookies](/blog/understanding-cookies)
Published: 2023-11-01
Category: Field Engineering
Learn how cookies function, how they are used by websites, and the importance of managing them for privacy and security.

### [How we optimized package imports in Next.js](/blog/how-we-optimized-package-imports-in-next-js)
Published: 2023-10-13
Category: Field Engineering
How solving barrel files led to faster cold boots and build times.

### [Tekla's ecommerce evolution: harnessing flexibility with Vercel and Medusa](/customers/teklas-ecommerce-evolution-harnessing-flexibility-with-vercel-and-medusa)
Published: 2023-10-11
Category: Customer stories
How Tekla uses Vercel and Medusa to provide speed, top performance, and flexibility to global ecommerce brands.

### [Announcing v0: Generative UI](/blog/announcing-v0-generative-ui)
Published: 2023-10-11
Category: v0
Introducing v0

### [Images on the web](/blog/images-on-the-web)
Published: 2023-10-10
Learn about the differences between the image formats JPEG, PNG, WebP, and AVIF with regards to compression techniques, resolution capabilities, pixel density, and effects on user experience for optimal web performance.

### [Introducing Spend Management](/blog/introducing-spend-management-realtime-usage-alerts-sms-notifications)
Published: 2023-10-05
Category: Company News
New controls to help you safely scale your projects and prevent unexpected bills.

### [Understanding the SameSite cookie attribute](/blog/understanding-the-samesite-cookie-attribute)
Published: 2023-10-02
Category: Field Engineering
Explore the SameSite cookie attribute's significance in ensuring web security and user privacy to strike the right balance between security and usability.

### [Understanding CSRF attacks](/blog/understanding-csrf-attacks)
Published: 2023-09-29
Understand the mechanics and risks of Cross-Site Request Forgery (CSRF) attacks, and discover crucial development practices, like anti-CSRF tokens and appropriate use of HTTP methods, to fortify web applications against such threats

### [First Input Delay (FID) vs. Interaction to Next Paint (INP)](/blog/first-input-delay-vs-interaction-to-next-paint)
Published: 2023-09-26
Learn about the differences between FID and INP and how to optimize your website's INP score.

### [Optimizing web fonts](/blog/optimizing-web-fonts)
Published: 2023-09-26
Learn how to optimize web fonts using resource hints, font-face descriptors, and the next/font module.

### [How Whop improved their Real Experience Score by 200% with the Next.js App Router](/customers/how-whop-improved-their-real-experience-score-by-200-with-the-next-js-app)
Published: 2023-09-21
Category: Customer stories
Whop improved developer experience with Vercel and Next.js

### [Why Vercel and Next.js are the perfect fit for this global fashion media group](/customers/why-vercel-and-next-js-are-the-perfect-fit-for-this-global-fashion-media)
Published: 2023-09-21
Category: Customer stories
l'officiel

### [Vercel achieves ISO 27001:2013 certification to further strengthen commitment to security ](/blog/vercel-iso-27001-security)
Published: 2023-09-12
Category: Security
Vercel achieves the ISO 27001 certification validating that our information security practices adhere to globally accepted standards.

### [How to create an optimal developer workflow](/blog/improving-developer-workflow)
Published: 2023-09-12
Category: Field Engineering
The way your team builds software matters: Create a developer experience that promotes iteration

### [How the at-home workout sensation, Hydrow, cut authoring times from weeks to minutes](/customers/hydrow)
Published: 2023-09-11
Category: Customer stories
Hydrow gains a seamless publishing workflow. Learn how this team succeeded in their mission to create the ultimate composable ecommerce workflow.

### [Using Zig in our incremental Turborepo migration from Go to Rust](/blog/how-we-continued-porting-turborepo-to-rust)
Published: 2023-09-08
Category: Field Engineering
In this Turborepo migration update, we explore the innovative strategies, such as the "Go Sandwich" approach and leveraging Zig's cross-compilation, guiding our gradual shift from Go to Rust, aiming to boost performance without sacrificing stability.

### [Why all application migrations should be incremental ](/blog/incremental-migrations)
Published: 2023-08-30
Category: Field Engineering
Projects that require migrations should aim for incremental migrations

### [Deploying at the speed of on-demand streaming](/customers/deploying-at-the-speed-of-on-demand-streaming)
Published: 2023-08-23
Category: Customer stories
Streamlined deployment process and superior developer experience for Joyn

### [Vercel AI Accelerator Demo Day](/blog/vercel-ai-accelerator-demo-day)
Published: 2023-08-23
Category: Company News
Watch 28 talented AI startups show off impressive demos in 3 minutes each as part of our AI Accelerator Demo Day.

### [Developing at the speed of sound: How Sonos amplified their DevEx](/customers/how-sonos-amplified-their-devex)
Published: 2023-08-17
Category: Customer stories
By switching to Vercel, Sonos leveled up their headless Next.js stack, cutting build times by 75%.

### [Konabos empowers an industry giant to deploy 50% faster with a composable stack ](/customers/konobos-empowers-industry-giant-to-deploy-50-faster)
Published: 2023-08-15
Category: Customer stories
American Bath Group's success with Konobos and Vercel

### [Algolia cuts build times in half with ISR using Next.js on Vercel](/blog/algolia-cuts-build-times-in-half-with-isr-using-next-js-on-vercel)
Published: 2023-08-09
How Algolia cuts build times in half with ISR using Next.js on Vercel

### [Introducing Next.js Commerce 2.0](/blog/introducing-next-js-commerce-2-0)
Published: 2023-08-07
Category: Company News
A high-performance ecommerce template, including support for BigCommerce, Medusa, Saleor, Shopify, and Swell.

### [Understanding React Server Components](/blog/understanding-react-server-components)
Published: 2023-08-01
Category: Field Engineering
React Server Components are changing the fundamental paradigms of React. Learn how Next.js handles the complexities and improves the performance of your applications.

### [Engineering a site at the speed of breaking news](/customers/washington-post-next.js-vercel-engineering-at-the-speed-of-breaking-news)
Published: 2023-07-27
Category: Customer stories
The Washington Post paid off tech debt by migrating to Next.js and Vercel between primary season and the 2022 US midterm elections. 

### [Introducing React Tweet](/blog/introducing-react-tweet)
Published: 2023-07-25
Category: Community
Embed tweets into your React application without sacrificing performance.

### [How Vercel helped this popular health database increase free trials by 284%](/customers/examine)
Published: 2023-07-25
Category: Customer stories
Examine

### [How Turborepo is porting from Go to Rust](/blog/how-turborepo-is-porting-from-go-to-rust)
Published: 2023-07-21
Category: Field Engineering
Our strategy for making updates and maintaining stability while we migrate languages.

### [How React 18 Improves Application Performance](/blog/how-react-18-improves-application-performance)
Published: 2023-07-19
Category: Field Engineering
Learn how React 18's concurrent features like Transitions, Suspense, and React Server Components improve application performance.

### [Iterating from design to deploy: the shape of future builders](/blog/iterating-from-design-to-deploy)
Published: 2023-07-13
Category: Community
A reflection on Guillermo Roach's Config 2023 talk, "The shape of future builders: from design to deploy."

### [Meet the Vercel AI Accelerator Participants](/blog/ai-accelerator-participants)
Published: 2023-07-12
Category: Company News
Meet the 40 startups and builders of the Vercel AI Accelerator.

### [Introducing the Vercel Platforms Starter Kit](/blog/platforms-starter-kit)
Published: 2023-07-05
Category: Company News
A fullstack template for building multi-tenant applications with custom domains using Next.js App Router, Vercel Postgres, and the Vercel Domains API.

### [Expanding the experimentation ecosystem with Edge Config and LaunchDarkly](/blog/edge-config-and-launch-darkly)
Published: 2023-06-27
Category: Community
We're excited to announce a new LaunchDarkly integration to bring low latency, global feature flags to your favorite frontend framework.

### [Incrementally adopting Next.js at one of Europe's fastest growing brands](/customers/incrementally-adopting-next-js-at-one-of-europes-fastest-growing-brands)
Published: 2023-06-23
Category: Customer stories
reMarkable goes composable 

### [An Introduction to Streaming on the Web](/blog/an-introduction-to-streaming-on-the-web)
Published: 2023-06-22
Category: Field Engineering
Learn how web streams work, their advantages, streaming on Vercel, and tools built around web streams.

### [How Neo Financial cut time spent on infrastructure admin by 50%](/customers/neo-financial)
Published: 2023-06-22
Category: Customer stories
Neo Financial leverages Vercel's frontend cloud to enhance their web development process, boost performance, and meet industry security standards—all while saving on resources. 

### [Enhanced content management for your headless CMS](/blog/enhanced-content-management-for-headless-cmses)
Published: 2023-06-22
Category: Company News
Draft Mode and Visual Editing for better content management making it easier to see your latest content changes before they’re published.

### [Introducing Skew Protection](/blog/version-skew-protection)
Published: 2023-06-21
Category: Company News
Skew Protection from Vercel, a mechanism to eliminate version skew

### [New features for SvelteKit: Optimize your application with ease](/blog/feature-complete-sveltekit)
Published: 2023-06-20
Category: Community
Get the most out of your SvelteKit app with pre-route configuration, Incremental Static Regeneration, and data at the Edge.

### [From idea to acquisition: How Potion.so shipped 4,000+ sites on Vercel](/customers/from-idea-to-acqusition-how-potion-shipped-4k-sites-on-vercel)
Published: 2023-06-15
Category: Customer stories
The success of Potion's one-person team is a testament to the power of Vercel's Frontend Cloud toolkit. 

### [Introducing the Vercel AI SDK](/blog/introducing-the-vercel-ai-sdk)
Published: 2023-06-15
Category: Company News
An interoperable, streaming-enabled, edge-ready software development kit for AI apps built with React and Svelte.

### [Introducing Vercel's AI Accelerator](/blog/vercel-ai-accelerator)
Published: 2023-06-14
Category: Company News
A 6 week program with over $850k in credits from Vercel and top AI platforms.

### [Visual Editing meets Markdown](/blog/visual-editing-meets-markdown)
Published: 2023-06-06
Category: Community
TinaCMS adopts Visual Editing

### [Designing the Vercel virtual product tour](/blog/designing-the-vercel-virtual-product-tour)
Published: 2023-06-02
Category: Field Engineering
Learn how and why we designed the Vercel virtual product tour to address some of the most prominent needs in our marketing funnel.

### [Celebrating 10 Years of React](/blog/10-years-of-react)
Published: 2023-05-29
Category: Community
Congratulations to the React team for a decade of innovation.

### [Vercel + Sanity: Innovating on a faster, more collaborative Web](/blog/vercel-sanity-innovating-on-a-faster-collaborative-web)
Published: 2023-05-17
Category: Company News
Vercel and Sanity together deliver a composable, cloud-native stack with Visual Editing, ISR, and pre-built templates to future-proof your frontend without sacrificing content velocity or developer experience.

### [What does Vercel do? ](/blog/what-is-vercel)
Published: 2023-05-10
Category: Community
Technically newsletter founder Justin Gage explains how Vercel makes building web applications as easy as possible, powered by the frontend cloud.

### [Improved support for Nuxt on Vercel](/blog/nuxt-on-vercel)
Published: 2023-05-05
Category: Community
Nuxt on Vercel now supports KV for Redis, Incremental Static Regeneration, and more.

### [Authentication for the frontend cloud](/blog/authentication-for-the-frontend-cloud)
Published: 2023-05-05
Category: Community
Learn how Clerk is reimagining authentication to embrace the architecture of framework-defined infrastructure.

### [Visual Editing: Click-to-edit content for headless CMSes](/blog/visual-editing)
Published: 2023-05-03
Category: Company News
Visual Editing

### [Quality software at scale with Vercel Spaces](/blog/vercel-spaces)
Published: 2023-05-03
Category: Company News
Spaces from Vercel

### [Introducing Vercel Firewall and Vercel Secure Compute](/blog/vercel-security)
Published: 2023-05-02
Category: Security
Vercel Secure Compute and Vercel Firewall for enhanced protection of your applications on the frontend cloud

### [Introducing storage on Vercel](/blog/vercel-storage)
Published: 2023-05-01
Category: Company News
New to the Vercel dashboard: Vercel KV, Vercel Postgres, and Vercel Blob. Announcing serverless storage solutions available on Vercel.

### [Vercel Web Analytics is now generally available](/blog/vercel-web-analytics-is-now-generally-available)
Published: 2023-04-19
Category: Company News
Get detailed, first-party page views, traffic analytics

### [Building towards operational excellence at CORE Construction](/customers/core-construction)
Published: 2023-04-18
Category: Customer stories
CORE

### [Making Commerce-UI a trusted partner for global ecommerce brands](/customers/making-commerce-ui-a-trusted-partner-for-global-ecommerce-brands)
Published: 2023-04-14
Category: Customer stories
commerce ui

### [Incremental migration from WordPress for a dev-first approach](/customers/incremental-migration-from-wordpress-for-a-dev-first-approach)
Published: 2023-04-14
Category: Customer stories
Gearbox migrated from WordPress and Gatsby to Next.js on Vercel, using Edge Middleware for incremental adoption while improving security and streamlining their development workflow.

### [Containing multi-site management within a single codebase](/customers/wunderman-thompson-composable-workflow)
Published: 2023-04-12
Category: Customer stories
Discover how to create an efficient design system that streamlines the site creation process, inspired by Wunderman Thompson's work in managing hundreds of brands from a single codebase.

### [How Vercel helps mmm.page manage over 30,000 sites](/customers/how-vercel-helps-mmm-page-manage-over-30-000-custom-domains)
Published: 2023-04-07
Category: Customer stories
mmm.page 

### [Vercel Edge Config is now generally available](/blog/vercel-edge-config-is-now-generally-available)
Published: 2023-04-06
Category: Company News
A globally distributed data store for low latency experimentation.

### [Powering a serverless Web: Vercel joins AWS Marketplace](/blog/vercel-joins-aws-marketplace)
Published: 2023-04-05
Category: Company News
Powering a serverless Web: Vercel joins the AWS Marketplace

### [Managing major traffic spikes during ticket drops with Vercel](/customers/managing-major-traffic-spikes-during-ticket-drops-with-vercel)
Published: 2023-03-31
Category: Customer stories
Customer story about how Shotgun used Vercel infrastructure, instant rollbacks, and request-time routing to handle major traffic spikes during ticket drops.

### [Replacing Google Optimize with the Vercel Edge Network](/blog/vercel-edge-google-optimize)
Published: 2023-03-30
Category: Field Engineering
Historical guide for replacing Google Optimize with Vercel Edge Config and request-time routing, with current guidance for Routing Middleware and Vercel Functions.

### [Streaming for Serverless Node.js and Edge Runtimes with Vercel Functions](/blog/streaming-for-serverless-node-js-and-edge-runtimes-with-vercel-functions)
Published: 2023-03-28
Category: Field Engineering
Learn how Vercel enables streaming for serverless Node.js and Edge runtimes. HTTP streaming enables the server to incrementally send response data in smaller chunks to the client while generating the complete response.

### [Custom fonts without compromise using  Next.js and `next/font`](/blog/nextjs-next-font)
Published: 2023-03-28
Category: Field Engineering
`next/font` automatically self-hosts your custom fonts, preventing layout shift and significantly reducing needed code.

### [How to build zero-CLS A/B tests with Next.js and Vercel Edge Config](/blog/zero-cls-experiments-nextjs-edge-config)
Published: 2023-03-23
Category: Field Engineering
Historical guide to zero-CLS A/B testing with Next.js and Edge Config, updated with current Vercel guidance for Routing Middleware and Vercel Functions.

### [Remix without limits (historical)](/blog/vercel-remix-integration-with-edge-functions-support)
Published: 2023-03-22
Category: Community
Historical announcement for Remix on Vercel with streaming SSR and multi-runtime support, updated with current Vercel Functions guidance.

### [Framework-defined infrastructure](/blog/framework-defined-infrastructure)
Published: 2023-03-07
Category: Field Engineering
Framework defined infrastructure means programmatic framework understanding for automatic infrastructure provisioning—an evolution from Infrastructure as Code (IaC).

### [Why Turborepo is migrating from Go to Rust](/blog/turborepo-migration-go-rust)
Published: 2023-03-07
Category: Field Engineering
How we're migrating from Go to Rust for better alignment with our tools and work

### [Introducing Vercel Monitoring](/blog/introducing-monitoring)
Published: 2023-03-06
Category: Company News
Vercel Monitoring

### [Your guide to headless commerce](/blog/your-guide-to-headless-commerce)
Published: 2023-02-27
Category: Community
Curious about headless commerce? Learn all about going headless with a composable stack so you can boost developer velocity, conversions, and more.

### [Optimizing performance for over 6M monthly visitors at CruiseCritic ](/customers/a-better-developer-experience-makes-building-cruise-critic-more-efficient)
Published: 2023-02-24
Category: Customer stories
Major improvements to the developer experience means Cruise Critic builds software more efficiently. 

### [Vercel Data Cache: A progressive cache, integrated with Next.js](/blog/vercel-cache-api-nextjs-cache)
Published: 2023-02-23
Category: Field Engineering
Cache only part of your page as static data, while fully dynamically rendering the rest of your application, including accessing real-time and personalized data.

### [Moving from monolithic WordPress to composable gives Plenti total freedom](/customers/from-monolith-to-composable-equipping-a-financial-services-ipo)
Published: 2023-02-23
Category: Customer stories
Plenti migrated from WordPress to Next.js on Vercel with a one-developer team, transforming their brand experience ahead of their rebrand and IPO.

### [The Next.js SEO Playbook: Ranking higher with Next.js on Vercel](/blog/nextjs-seo-playbook)
Published: 2023-02-23
Category: Community
SEO is a critical priority in Next.js on Vercel, letting you rank higher in search engine results and better optimize your content for users.

### [How Makeswift improved CI speed by 65% with Turborepo](/customers/how-makeswift-improved-ci-speed-by-65-with-turborepo)
Published: 2023-02-22
Category: Customer stories
makeswift

### [Introducing Vercel Cron Jobs](/blog/cron-jobs)
Published: 2023-02-22
Category: Company News
Automate repetitive tasks using Vercel Cron Jobs and Vercel Functions. Available today in beta.

### [How a global agency built a web innovation engine in two months ](/customers/how-a-global-agency-built-a-web-innovation-engine-in-two-months)
Published: 2023-02-22
Category: Customer stories
Globacore

### [How Vercel and Next.js keep Rippling on their rising path to success](/customers/how-vercel-and-next-js-keep-rippling-on-their-rising-path-to-success)
Published: 2023-02-22
Category: Customer stories
How Rippling uses headless Wordpress and Vercel to iterate faster

### [How Indent delivers secure access with Next.js and Vercel](/customers/how-indent-delivers-secure-access-with-next.js-and-vercel)
Published: 2023-02-17
Category: Customer stories
Indent

### [Less code, better UX: Fetching data faster with the Next.js 13 App Router](/blog/nextjs-app-router-data-fetching)
Published: 2023-02-10
Category: Field Engineering
Fetching data in Next.js 13 has been vastly improved with Server Components, smarter caching, and Loading UI.

### [Runway enables next-generation content creation with AI and Vercel](/customers/runway-enables-next-generation-content-creation-with-ai-and-vercel)
Published: 2023-02-10
Category: Customer stories
Learn how Runway migrated to Vercel, resulting in faster build times, review cycles, and better performance.

### [From newsletter to global media brand with a frontend cloud](/customers/from-newsletter-to-global-media-brand-with-a-headless-frontend)
Published: 2023-02-09
Category: Customer stories
Morning brew

### [Navigating tradeoffs in large-scale website migrations](/blog/navigating-tradeoffs-in-large-scale-website-migrations)
Published: 2023-02-09
Category: Community
Navigating large scale migrations

### [Faster iteration with Turborepo and Vercel Remote Cache](/blog/vercel-remote-cache-turbo)
Published: 2023-02-07
Category: Field Engineering
Vercel Remote Cache makes your Turborepo caching strategy multiplayer, allowing you to share caches with your teammates and CI.

### [Super serves thousands of domains from a single codebase with Next.js and Vercel](/customers/super-serves-thousands-of-domains-on-one-project-with-next-js-and-vercel)
Published: 2023-02-01
Category: Customer stories
How Super uses Next.js and Vercel to power thousands of Notion-based websites with custom domains from a single codebase.

### [Building a GPT-3 app with Next.js and streaming (historical)](/blog/gpt-3-app-next-js-vercel-edge-functions)
Published: 2023-02-01
Category: Field Engineering
Historical tutorial for a GPT-3 Twitter bio generator. For new AI apps on Vercel, use Vercel Functions with Fluid compute and streaming.

### [Behind the scenes of Vercel's infrastructure: Achieving optimal scalability and performance](/blog/behind-the-scenes-of-vercels-infrastructure)
Published: 2023-01-27
Category: Field Engineering
Learn how Vercel builds, deploys, and scales serverless applications with speed and global reliability

### [How Plex 6x their impressions deploying Next.js on Vercel](/customers/how-plex-6x-their-impressions-deploying-next-js-on-vercel)
Published: 2023-01-26
Category: Customer stories
Plex customer story

### [Deploying AI-driven apps on Vercel](/blog/deploying-ai-applications)
Published: 2023-01-25
Category: Field Engineering
Deploying AI-driven apps on Vercel is easier than ever before. Here's some templates and tooling to jump-start your AI application.

### [How Supabase elevated their developer experience with Turborepo](/customers/how-supabase-elevated-their-developer-experience-with-turborepo)
Published: 2023-01-24
Category: Customer stories
Supabase and Turborepo

### [Improving readability with React Wrap Balancer](/blog/react-wrap-balancer)
Published: 2023-01-19
Category: Field Engineering
React Wrap Balancer tidies up bad typography and matches line lengths on the fly.

### [Delivering AI analysis faster with the Vercel workflow](/customers/delivering-ai-analysis-faster-with-the-vercel-workflow)
Published: 2023-01-17
Category: Customer stories
Viable customer story

### [How Vercel enables Wunderman Thompson to launch global brands](/customers/how-vercel-enables-wunderman-thompson-to-launch-global-brands)
Published: 2023-01-17
Category: Customer stories
Launching a global brand with Vercel and Wunderman Thompson, unlocking the potential of international brands through strategic, digital-led growth.

### [Sanity balances experimentation and performance with Vercel Edge Middleware ](/customers/sanity-edge-middleware)
Published: 2023-01-13
Category: Customer stories
Setting the standard for A/B testing and performance with Vercel Edge Middleware.

### [Edge Functions enable Read.cv to deliver profiles globally, with near-zero latency](/customers/edge-functions-enable-read-cv-to-deliver-profiles-globally-with-near-zero)
Published: 2023-01-13
Category: Customer stories
Customer story about Read.cv using Vercel to deliver personalized profile pages with low latency and flexible custom-domain infrastructure.

### [Hashnode runs the fastest blogs on the web with Vercel](/customers/hashnode-runs-the-fastest-blogs-on-the-web-with-vercel)
Published: 2023-01-13
Category: Customer stories
Hashnode

### [Helping Swell’s merchants provide unparalleled ecommerce experiences](/customers/helping-swells-merchants-provide-unparalleled-ecommerce-experiences)
Published: 2023-01-13
Category: Customer stories
Swell

### [Vercel + Sitecore: Partnering on a composable future](/blog/vercel-sitecore-partnership)
Published: 2023-01-12
Category: Company News
Sitecore and Vercel Partnership

### [The Turbopack vision](/blog/the-turbopack-vision)
Published: 2023-01-11
Category: Community
Watch webpack founder Tobias Koppers talk about the vision for Turbopack at React Day Berlin.

### [Building a global streetwear label with Next.js](/blog/kidsuper-innovates-with-next.js)
Published: 2023-01-10
Category: Community
Next.js and Vercel allow KidSuper to easily scale and reflect the brand's creativity online. 

### [Building a fast, animated image gallery with Next.js](/blog/building-a-fast-animated-image-gallery-with-next-js)
Published: 2023-01-09
Category: Field Engineering
Learn how to build a performant image gallery using the Next.js image component and Cloudinary that can handle hundreds of large images and deliver great UX.

### [Turbocharging Next.js: How Remote Caching decreased publish times by 80%](/blog/turborepo-remote-cache-nextjs-publish-times-80-percent)
Published: 2022-12-22
Category: Field Engineering
The Next.js release process got 80% faster using Turborepo Remote Caching.

### [How to optimize your Next.js site: Tips from industry leaders](/blog/optimize-your-nextjs-site)
Published: 2022-12-21
Category: Community
Hear from industry leaders on how you can best use React Server Components, the latest in Web UI, powerful layouts, and more to create a fast site.

### [Enhanced Preview experience](/blog/making-live-reviews-a-reality-enhanced-preview-experience)
Published: 2022-12-20
Category: Company News
Vercel Preview Deployments

### [Vercel at AfroTech 2022: An immersive experience](/blog/vercel-at-afrotech-2022)
Published: 2022-12-19
Category: Company News
Last month, Vercel had the privilege of sponsoring AfroTech Conference 2022. This was our approach. 

### [Deployment Protection: Added security controls now available on all plans](/blog/protecting-deployments)
Published: 2022-12-19
Category: Security
Deployment protection on all plans - Secure your deployment previews and create shareable links to share with collaborators

### [Building a powerful notification system for Vercel with Knock](/blog/building-a-powerful-notification-system-for-vercel-with-knock-app)
Published: 2022-12-16
Category: Community
Notifications on Vercel

### [Introducing Edge Config: Globally distributed, instant configuration](/blog/edge-config-public-beta)
Published: 2022-12-15
Category: Company News
Historical announcement for Edge Config public beta, updated with current guidance for reading configuration from Vercel Functions and Routing Middleware.

### [Vercel Edge Functions are now generally available](/blog/edge-functions-generally-available)
Published: 2022-12-15
Category: Company News
Historical announcement for the Edge Functions GA release, updated with current guidance for Vercel Functions, Fluid compute, and the Edge Runtime.

### [Announcing SvelteKit Auth: Bringing NextAuth.js to all frameworks](/blog/announcing-sveltekit-auth)
Published: 2022-12-14
Category: Community
SvelteKit Auth is a simple, low configuration authentication library for SvelteKit applications, with support for many popular OAuth providers.

### [Using SvelteKit 1.0 on Vercel](/blog/using-sveltekit-1-0-on-vercel)
Published: 2022-12-14
Category: Community
SvelteKit is a full-stack framework built on the Svelte compiler that provides directory-based routing, SSR, static site generation, and zero-configuration deployment on Vercel for fast, lightweight web applications.

### [From idea to 100 million views: Building a viral application for your personal music festival](/blog/from-idea-to-100-million-views-instafest-music-festival-application)
Published: 2022-12-12
Category: Community
Learn how Anshay Saboo, a Computer Science student at USC, used Next.js and Vercel to launch Instafest fast and scale to 500,000 new users per hour.

### [Migrating a large, open-source React application to Next.js and Vercel](/blog/migrating-a-large-open-source-react-application-to-next-js-and-vercel)
Published: 2022-12-08
Category: Field Engineering
Learn how we approached migrating the BBC's large, open-source React application to Next.js and Vercel to see both developer and user experience benefits.

### [AWS and Vercel: Accelerating innovation with serverless computing](/blog/aws-and-vercel-accelerating-innovation-with-serverless-computing)
Published: 2022-12-06
Category: Company News
Vercel's edge-first serverless platform accelerates development with Preview Deployments and delivers production performance at scale, as proven by The Washington Post's seamless US Midterm Elections coverage.

### [DatoCMS builds 60% faster with a streamlined workflow](/customers/datocms-builds-60-faster-with-a-streamlined-workflow)
Published: 2022-11-30
Category: Customer stories
Building a smarter frontend workflow with Next.js on Vercel 

### [How Scale AI unifies design and performance with Next.js and Vercel](/customers/scale-unifies-design-and-performance-with-next-js-and-vercel)
Published: 2022-11-30
Category: Customer stories
How Scale is unlocking faster design and development at scale with Vercel and Next.js, while improving performance

### [How Vercel helped justInCase Technologies cut their build time in half](/customers/how-vercel-helped-justincase-technologies-cut-their-build-time-in-half)
Published: 2022-11-30
Category: Customer stories
How justInCase Technologies saves 72 hours of dev time per month with Vercel

### [With Next.js, Vercel, and Sanity, Loom empowers every team to iterate](/customers/loom-headless-with-nextjs)
Published: 2022-11-30
Category: Customer stories
By going headless with Next.js on Vercel, Loom ensures the best experience for both their developers and cu

### [Edge Config: Ultra-low latency data at the edge](/blog/edge-config-ultra-low-latency-data-at-the-edge)
Published: 2022-11-23
Category: Field Engineering
Historical announcement for Edge Config, a globally replicated data store for low-latency reads from Vercel Functions and Routing Middleware.

### [Using Vercel comments to improve the Next.js 13 documentation](/blog/using-vercel-comments-to-improve-the-next-js-13-documentation)
Published: 2022-11-03
Category: Community
Help us build the Next.js 13 documentation by commenting directly on the docs themselves.

### [Introducing Turbopack](/blog/turbopack)
Published: 2022-10-25
Category: Company News
Introducing Turbopack, the Rust-based successor to Webpack. A high-performance bundler for React Server Components and TypeScript codebases.

### [Vercel acquires Splitbee to expand first-party analytics](/blog/vercel-acquires-splitbee)
Published: 2022-10-25
Category: Company News
Introducing the next generation of Vercel Analytics

### [Building an interactive WebGL experience in Next.js](/blog/building-an-interactive-webgl-experience-in-next-js)
Published: 2022-10-21
Category: Field Engineering
Use interactive code sandboxes to build the Next.js Conf registration prism game.

### [Regional execution for ultra-low latency rendering at the edge](/blog/regional-execution-for-ultra-low-latency-rendering-at-the-edge)
Published: 2022-10-20
Category: Field Engineering
Historical post about regional execution for edge compute, updated with current guidance for configuring Vercel Function regions near your data source.

### [Next.js Conf 2022: Iterate, scale, and deliver a great UX](/blog/nextjs-conf-2022-iterate-scale-deliver)
Published: 2022-10-18
Category: Community
Learn how to iterate, scale, and deliver a great UX from the experts speaking at this year’s Next.js Conf. Register for the online event today.

### [Introducing OG Image Generation: Fast, dynamic social card images at the Edge](/blog/introducing-vercel-og-image-generation-fast-dynamic-social-card-images)
Published: 2022-10-10
Category: Company News
Announcing Vercel OG Image Generation, a new library for generating dynamic social card images.

### [Improving the accessibility of our Next.js site](/blog/improving-the-accessibility-of-our-nextjs-site)
Published: 2022-09-30
Category: Field Engineering
Let's make the Web. Accessible.

### [How the world’s biggest YouTuber served millions of users on Vercel](/blog/serving-millions-of-users-on-the-new-mrbeast-storefront)
Published: 2022-09-29
Category: Community
Find out how basement.studio balanced performance, entertainment, and reliability for MrBeast's new storefront. 

### [Introducing Commenting on Preview Deployments](/blog/introducing-commenting-on-preview-deployments)
Published: 2022-09-22
Category: Company News
Commenting in Vercel Preview Deployment

### [Next.js Layouts RFC in 5 minutes](/blog/next-js-layouts-rfc-in-5-minutes)
Published: 2022-09-14
Category: Field Engineering
Learn about the upcoming routing and layouts changes to Next.js.

### [Using the latest Next.js 12.3 features on Vercel](/blog/using-the-latest-next-js-12-3-features-on-vercel)
Published: 2022-09-13
Category: Field Engineering
Vercel natively supports and extends Next.js 12.3, allowing teams to improve their workflow and iterate faster.

### [Building a viral application to visualize train routes](/blog/building-a-viral-application-to-visualize-train-routes)
Published: 2022-09-10
Category: Community
How Benjamin Td built a viral application called Chronotrains to visualize train routes across Europe.

### [Introducing the Vercel Templates Marketplace](/blog/introducing-the-vercel-templates-marketplace)
Published: 2022-09-09
Category: Company News
We are excited to announce the launch of the Vercel Templates Marketplace.

### [Curve fitting for charts: better visualizations for Vercel Analytics](/blog/curve-fitting-for-charts-better-visualizations-for-vercel-analytics)
Published: 2022-09-09
Category: Field Engineering
How we made your Vercel Analytics data more actionable to drive performance for your application.

### [How to run A/B tests with Next.js and Vercel](/blog/ab-testing-with-nextjs-and-vercel)
Published: 2022-09-09
Category: Field Engineering
Learn how to run A/B tests with Next.js and Vercel Edge Middleware.

### [At Next.js Conf 2022, learn to build better and scale faster](/blog/nextjs-conf-2022)
Published: 2022-09-02
Category: Community
Next.js Conf 2022: Presented by Vercel

### [How SZA and Integral Studio create at the moment of inspiration](/blog/sza-integral-create-at-the-moment-of-inspiration)
Published: 2022-08-29
Category: Community
In 2017, Integral Studio chose Vercel & Next.js to create a site that would mirror the creativity of then-up-and-coming artist, SZA. To celebrate the 5-year anniversary and re-release of her Grammy-winning album, they once again turned to Vercel.

### [Introducing support for WebAssembly at the Edge](/blog/introducing-support-for-webassembly-at-the-edge)
Published: 2022-08-26
Category: Field Engineering
Historical announcement for WebAssembly support at the edge, updated with current guidance for Vercel Functions and the WebAssembly runtime reference.

### [How we made the Vercel Dashboard twice as fast](/blog/how-we-made-the-vercel-dashboard-twice-as-fast)
Published: 2022-08-09
Category: Field Engineering
Let’s review the techniques and strategies we used to improve the Vercel Dashboard so you can make a data-driven impact on your application. 

### [Improving INP with React 18 and Suspense](/blog/improving-interaction-to-next-paint-with-react-18-and-suspense)
Published: 2022-08-09
Category: Field Engineering
Learn how to improve Interaction to Next Paint in React applications by using Suspense and selective hydration.

### [Hashnode runs the fastest blogs on the web with Vercel](/customers/hashnode-runs-faster-blogs-on-the-web-with-vercel)
Published: 2022-08-03
Category: Customer stories
After evaluating alternative solutions like AWS Amplify, Hashnode ultimately chose Vercel because of the ability to manage custom domains at scale and the smooth and intuitive developer experience.

### [Build your own web framework](/blog/build-your-own-web-framework)
Published: 2022-07-28
Category: Field Engineering
Historical tutorial for building a web framework on Vercel, updated with current Vercel Functions and Routing Middleware terminology.

### [Announcing the Build Output API](/blog/build-output-api)
Published: 2022-07-21
Category: Company News
The Build Output API enables any framework, including your own custom-built solution, to take advantage of Vercel’s infrastructure building blocks.

### [Vercel Edge Middleware: Dynamic at the speed of static (historical)](/blog/vercel-edge-middleware-dynamic-at-the-speed-of-static)
Published: 2022-06-28
Category: Company News
Execute custom logic at the moment of request, pushing personalization to the edge with exceptional performance.

### [Introducing the Edge Runtime](/blog/introducing-the-edge-runtime)
Published: 2022-06-21
Category: Field Engineering
To enable every framework to build for the edge, we’re releasing edge-runtime: a toolkit for developing, testing, and defining the runtime web APIs for edge infrastructure.

### [MongoDB and Vercel: from idea to global fullstack app in seconds](/blog/mongodb-and-vercel-from-idea-to-global-fullstack-app-in-seconds)
Published: 2022-06-13
Category: Community
At this year's MongoDB World, we announced the MongoDB and Vercel integration—and shared our vision for enabling developers to create at the moment of inspiration. Let’s explore how MongoDB and Vercel make that possible. 

### [How HashiCorp developers iterate faster with Incremental Static Regeneration](/blog/how-hashicorp-developers-iterate-faster-with-isr)
Published: 2022-04-26
Category: Field Engineering
HashiCorp used ISR and on-demand ISR in Next.js 12 to cut build times, update pages instantly, and scale docs across 8 products without full site rebuilds.

### [Upgrading Next.js for instant performance improvements](/blog/upgrading-nextjs-for-instant-performance-improvements)
Published: 2022-03-17
Category: Field Engineering
Learn how Next.js provides a toolkit to improve site performance, improve the developer experience, and decrease build times with every upgrade.

### [Monorepos are changing how teams build software](/blog/monorepos)
Published: 2022-03-03
Category: Community
Monorepos are codebases containing multiple projects in a single unified code repository. This post explores how monorepos can improve your development workflow.

### [The evolution of the Web: What we learned and where we’re going](/blog/how-the-web-evolves)
Published: 2022-02-02
Category: Company News
From open source to a more powerful edge, see our predictions for the future of frontend development—featuring experts in React, Next.js, Svelte, and more. 

### [The future of Svelte, an interview with Rich Harris](/blog/the-future-of-svelte-an-interview-with-rich-harris)
Published: 2021-12-15
Category: Community
In this 45-minute interview, hear Rich Harris (the creator of Svelte) talk about its plans for the future. Other topics include funding open-source, SvelteKit 1.0, the Edge-first future, and more.

### [Supporting the Future of React](/blog/supporting-the-future-of-react)
Published: 2021-12-14
Category: Company News
An update on our support for React and other open-source libraries our customers depend on.

### [Vercel acquires Turborepo to accelerate build speed and improve developer experience ](/blog/vercel-acquires-turborepo)
Published: 2021-12-09
Category: Company News
Vercel acquires Turborepo to accelerate build speed and improve developer experience

### [Announcing $150M to build the end-to-end platform for the modern Web](/blog/vercel-funding-series-d-and-valuation)
Published: 2021-11-23
Category: Company News
Vercel raised a $150M Series D at a $2.5B valuation to accelerate its mission to make the Web faster for developers and teams.

### [Vercel welcomes Rich Harris, creator of Svelte  ](/blog/vercel-welcomes-rich-harris-creator-of-svelte)
Published: 2021-11-11
Category: Company News
We're excited to share Rich Harris, the creator of Svelte, has joined Vercel to make the Web. Faster.

### [At Next.js Conf 2021, let’s make the Web. Faster.](/blog/at-next-js-conf-2021-lets-make-the-web-faster)
Published: 2021-09-20
Category: Company News
Join us for Next.js Conf on October 26, 2021. 

### [Welcoming Kathy Korevec to Vercel, our new Head of Product](/blog/welcoming-kathy-korevec-to-vercel-our-new-head-of-product)
Published: 2021-07-07
Category: Company News
We’re excited to announce that Kathy Korevec is joining our leadership team at Vercel as Head of Product to accelerate our mission to bring the best developer experience to our customers and community.

### [Supercharge your Vercel Projects with Integrations](/blog/integrations-marketplace)
Published: 2021-07-01
Category: Community
Connect your Vercel project to databases, monitoring tools, commerce providers, developer tools, and more.

### [$102M to Continue Building the Next Web, Together](/blog/series-c-102m-continue-building-the-next-web)
Published: 2021-06-23
Category: Company News
Vercel has added an additional $102 million of investment at a valuation greater than $1BN.

### [Next.js 11, Next.js Live and more: A recap of Next.js Conf Special Edition](/blog/nextjs-special-event-recap)
Published: 2021-06-22
Category: Company News
Learn more about the latest releases announced at Next.js Conf, including Next.js 11 and Next.js Live.

### [How Core Web Vitals Will Impact Google Rankings in 2021](/blog/core-web-vitals)
Published: 2021-04-15
Category: Field Engineering
Landing a top spot on Google can have a multi-million dollar impact on your business. Starting in June 2021, the performance of your site (determined by Core Web Vitals) will be critical to your search ranking.

### [Nuxt Analytics on Vercel](/blog/nuxt-analytics-on-vercel)
Published: 2021-02-26
Category: Community
Starting today, Vercel Analytics is available for Nuxt projects – without any configuration.

### [Visualize Team Usage With Sophisticated Usage Dashboard](/blog/sophisticated-usage-dashboard)
Published: 2021-02-23
Category: Company News
Learn more about how Vercel's latest feature will give access to usage information so your Team can optimize your builds.

### [Vercel & Next.js Experts Help Teams Build the Next Big Thing](/blog/vercel-and-next-js-experts-help-teams-build-the-next-big-thing)
Published: 2021-02-16
Category: Company News
Learn more about how to collaborate, build, and succeed with one of Vercel's Agency Experts, or become an Agency Expert today. 

### [Transfer Vercel projects with zero downtime](/blog/transfer-vercel-projects-with-zero-downtime)
Published: 2021-01-28
Category: Company News
With Vercel, you can now transfer projects from Hobby plan to a Vercel Team Plan with Zero Downtime

### [10 Next.js tips you might not know](/blog/10-next-js-tips-you-might-not-know)
Published: 2021-01-26
Category: Community
Discover 10 expert Next.js tips including redirects, rewrites, preview mode, API routes, and performance optimizations to boost your development workflow. 

### [React Server Components with Next.js](/blog/everything-about-react-server-components)
Published: 2021-01-15
Category: Field Engineering
Learn about React Server Components (experimental) and how they'll change how we build React applications, creating a better end-user experience.

### [Three Improvements to Project Creation & Git Integration](/blog/three-improvements-to-vercel-project-creation-vercel-git-integration)
Published: 2020-12-18
Category: Company News
We've improved the developer experience by introducing three updates that apply to all users on Hobby, Pro, and Enterprise plans.

### [$40M to Build the Next Web](/blog/series-b-40m-to-build-the-next-web)
Published: 2020-12-16
Category: Company News
Today we announce $40M in new funding to help everyone build the next web.

### [Vercel Analytics for Gatsby](/blog/gatsby-analytics)
Published: 2020-11-04
Category: Community
Starting today, Vercel Analytics is available for Gatsby projects – without any configuration.

### [September 2020](/blog/changelog-september-2020)
Published: 2020-09-01
Category: Company News
Vercel's changelog for September 2020

### [Monorepos](/blog/monorepos-are-changing-how-teams-build-software)
Published: 2020-08-28
Category: Company News
For greater collaboration and flexibility at scale, Vercel now supports monorepos.

### [August 2020](/blog/changelog-august-2020)
Published: 2020-08-01
Category: Company News
Vercel's changelog for August 2020

### [Our new Edge and Dev infrastructure](/blog/new-edge-dev-infrastructure)
Published: 2020-07-21
Category: Company News
Introducing major end-to-end enhancements from a better development experience to serving pages even faster.

### [Custom production branch](/blog/custom-production-branch)
Published: 2020-07-17
Category: Company News
As of today, you can customize the Production Branch of your Projects right in the Project Settings.

### [Next.js: Server-side Rendering vs. Static Generation](/blog/nextjs-server-side-rendering-vs-static-generation)
Published: 2020-07-09
Category: Field Engineering
How to use Static Generation, Incremental Static Generation, and Client-side Fetching with Next.js.

### [July 2020](/blog/changelog-july-2020)
Published: 2020-07-01
Category: Company News
Vercel's changelog for July 2020

### [DNS Records UI](/blog/dns-records-ui)
Published: 2020-06-23
Category: Company News
Configure custom DNS Records for your Domains and apply presets. Right in the Web UI.

### [June 2020](/blog/changelog-june-2020)
Published: 2020-06-01
Category: Company News
Vercel's changelog for June 2020

### [May 2020](/blog/changelog-may-2020)
Published: 2020-05-01
Category: Company News
Vercel's changelog for May 2020

### [Protecting Deployments](/blog/security-controls-protected-preview-deployments-passwords)
Published: 2020-05-01
Category: Company News
Enable Password or SSO Protection to restrict access to your Deployments.

### [ZEIT is now Vercel](/blog/zeit-is-now-vercel)
Published: 2020-04-21
Category: Company News
Today, we have some very special news regarding the evolution of our company.

### [Environment Variables UI](/blog/environment-variables-ui)
Published: 2020-04-14
Category: Company News
Configure different Environment Variables for Production, Preview, and Development – right in the Dashboard.

### [Simpler Pricing](/blog/simpler-pricing)
Published: 2020-04-08
Category: Company News
With our new Hobby, Pro, and Enterprise plans, all your needs are covered.

### [April 2020](/blog/changelog-april-2020)
Published: 2020-04-01
Category: Company News
Vercel's changelog for April 2020

### [We're All in This Together](/blog/we-are-all-in-this-together)
Published: 2020-03-25
Category: Company News
A showcase of projects from the Vercel developer community, built during the COVID-19 pandemic.

### [Canceling Ongoing Deployments](/blog/canceling-ongoing-deployments)
Published: 2020-03-24
Category: Company News
Preventing ongoing deployments from building is now simply a matter of clicking a button.

### [New Git Integration Settings](/blog/new-git-integration-settings)
Published: 2020-03-23
Category: Company News
 Thanks to an overhauled UI, managing the Git connection of a project or an entire account is easier than ever.

### [Refined Logging](/blog/refined-logging)
Published: 2020-03-11
Category: Company News
The refined UI for Build and Serverless Function logs makes consuming logs a pleasure.

### [March 2020](/blog/changelog-march-2020)
Published: 2020-03-01
Category: Company News
Vercel's changelog for March 2020

### [Advanced Project Settings](/blog/advanced-project-settings)
Published: 2020-02-06
Category: Company News
Fully customize your project’s behaviour or hit the ground running using our new framework presets.

### [Get support from the dashboard](/blog/support-form)
Published: 2020-02-03
Category: Company News
Get in touch with Vercel Support easier than ever before, without having to leave your dashboard.

### [February 2020](/blog/changelog-february-2020)
Published: 2020-02-01
Category: Company News
Vercel's changelog for February 2020

### [Log Drains](/blog/log-drains)
Published: 2020-01-31
Category: Company News
Easily forward and collect your logs using Log Drains.

### [January 2020](/blog/changelog-january-2020)
Published: 2020-01-01
Category: Company News
Vercel's changelog for January 2020

### [backendlessConf_ 2019](/blog/our-first-online-conference)
Published: 2019-12-23
Category: Company News
Catch all the highlights and important links from the first-ever backendlessConf_ representing the frontend and JAMstack.

### [Branch Domains](/blog/branch-domains)
Published: 2019-12-20
Category: Company News
Assign a Git Branch to your domain, so that every deployment created on it will automatically receive the domain.

### [December 2019](/blog/changelog-december-2019)
Published: 2019-12-01
Category: Company News
Vercel's changelog for December 2019

### [Vercel for Bitbucket](/blog/bitbucket)
Published: 2019-11-27
Category: Company News
Push code to Bitbucket and automatically deploy with Vercel.

### [Dashboard redesign](/blog/dashboard-redesign)
Published: 2019-11-20
Category: Company News
We're bringing the simplicity of our developer experience to our web dashboard. Creating new projects, importing existing code, managing domains, setting up redirects, inspecting deployments and functions, and managing teams has never been easier.

### [Introducing the Deploy Button](/blog/deploy-button)
Published: 2019-11-18
Category: Company News
Make your project deployable with the click of a button.

### [Inspecting Serverless Functions](/blog/functions-tab)
Published: 2019-11-18
Category: Company News
 Get insight into your Serverless Functions with the new "Functions" dashboard tab.

### [Customizing Serverless Functions](/blog/customizing-serverless-functions)
Published: 2019-11-12
Category: Company News
With the new `functions` property, you can configure your serverless functions.

### [November 2019](/blog/changelog-november-2019)
Published: 2019-11-01
Category: Changelog
Vercel's changelog for November 2019

### [Default Production Domain](/blog/default-production-domain)
Published: 2019-10-31
Category: Company News
Every new project now receives a default production domain.

### [Redirecting Domains](/blog/redirecting-domains)
Published: 2019-10-29
Category: Company News
As of today, you can redirect your domains to each other.

### [Advanced Invoice Settings](/blog/advanced-invoice-settings)
Published: 2019-10-02
Category: Company News
 As of today, invoices received from Vercel can be extended with much more information. 

### [October 2019](/blog/changelog-october-2019)
Published: 2019-10-01
Category: Company News
Vercel's changelog for October 2019

### [Introducing Wildcard Domains](/blog/wildcard-domains)
Published: 2019-09-10
Category: Company News
Wildcard Domains allow for pointing all imaginable sub domains to a project, without having to define those sub domains at all.

### [Deploy Summary Integration](/blog/deploy-summary)
Published: 2019-09-03
Category: Company News
Introducing Deploy Summary, a Vercel integration to enhance your Pull Requests or Merge Requests with screenshots and links to your changed pages.

### [Zero Config Deployments](/blog/zero-config)
Published: 2019-08-07
Category: Company News
Deploy frontends and serverless functions without any configuration.

### [Introducing Deploy Hooks](/blog/introducing-deploy-hooks)
Published: 2019-07-30
Category: Company News
With Deploy Hooks, you can create a deployment based on any event.

### [Node.js 10 is Now Available](/blog/node-10)
Published: 2019-06-25
Category: Community
We are enabling Node.js 10 support for new serverless Node.js functions and Next.js applications deployed with Vercel.

### [Helpers for Serverless Node.js Functions](/blog/vercel-node-helpers)
Published: 2019-06-19
Category: Community
Introducing six default methods to the request and response payloads in your Node.js Serverless Functions.

### [Vercel Hackathon Winners](/blog/hackathon-winners)
Published: 2019-06-07
Category: Company News
The first-ever Vercel Hackathon was a phenomenal success. Read on to learn about winners, and a summary of the event.

### [Windows Support for `vercel dev`](/blog/vercel-dev-windows)
Published: 2019-05-07
Category: Company News
As of today, you can use the `vercel dev` command of Vercel CLI on Windows.

### [Introducing Serverless Pre-Rendering (SPR)](/blog/serverless-pre-rendering)
Published: 2019-05-03
Category: Company News
Introducing SPR, an industry-defining feature that allows you to get the best of both static and dynamic data rendering.

### [Introducing `vercel dev`: Serverless, on localhost](/blog/vercel-dev)
Published: 2019-04-30
Category: Company News
With Vercel CLI's new `vercel dev` command, you can locally work on Vercel applications easily, without having to deploy them.

### [Automatic SSL with Vercel and Let's Encrypt](/blog/automatic-ssl-with-vercel-lets-encrypt)
Published: 2019-04-16
Category: Company News
Learn how Vercel uses Let's Encrypt to provision free SSL certificates for all users, automatically.

### [Auto Job Cancellation for Vercel for GitHub](/blog/auto-job-cancellation-for-vercel-github)
Published: 2018-11-15
Category: Company News
Deploying the latest push with Vercel for GitHub for the latest changes in an instant.

### [Next.js 6.1](/blog/next6-1)
Published: 2018-06-27
Category: Community
Next.js 6.1 features improved reliability and consistency in development.

### [Next.js 6 and Nextjs.org](/blog/next6)
Published: 2018-05-16
Category: Community
Next.js 6 features zero-configuration static exports, App Component, Babel 7 and more

### [Next.js 5.1: Faster Page Resolution](/blog/next5-1)
Published: 2018-03-26
Category: Community
Next.js 5.1 features support for environment configuration, phases, source maps, and new Next.js plugins.

### [Next.js 5: Universal Webpack, CSS Imports, Plugins and Zones](/blog/next5)
Published: 2018-02-05
Category: Community
Next.js 5 focuses on greater extensibility, composability for large applications and performance

### [Towards Next.js 5: Introducing Canary Updates](/blog/next-canary)
Published: 2017-11-15
Category: Community
Featuring a new update channel for Next.js, our first canary release and the Next.js 5 Roadmap.

### [Next.js 4: React 16 and styled-jsx 2](/blog/next4)
Published: 2017-10-09
Category: Community
Next.js 4: React 16 and styled-jsx 2

### [Next.js 3.0](/blog/next3)
Published: 2017-08-08
Category: Community
Next.js 3.0 comes with vastly improved HMR, dynamic imports, static exports and better serverless support!

### [Next 3.0 Preview: Static Exports and Dynamic Imports](/blog/next3-preview)
Published: 2017-05-15
Category: Community
Next 3.0 features Static Exports with one command and Dynamic Imports

### [Next.js 2.0](/blog/next2)
Published: 2017-03-27
Category: Community
Next.js 2.0 comes packed with performance improvements and extensibility features

### [Next.js](/blog/next)
Published: 2016-10-25
Category: Community
We're very proud to open-source Next.js, a small framework for server-rendered universal JavaScript webapps.

---

## Related

- [Changelog](/changelog/sitemap.md)
- [Pricing](/pricing)
- [Documentation](https://vercel.com/docs)
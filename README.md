NASA EVA Command-Action Dataset

This repository contains a comprehensive dataset of 20,000+ dialogue lines from 7 major Apollo missions (11, 12, 13, 14, 15, 16, 17), classified into structured Command-Action pairs for training AI models.

Dataset Structure 

The data is organized by semantic classification category, stored as individual JSONL files directly in the Data folder:

- navigation_commands.jsonl: Commands related to movement, heading, or traversing (e.g., "Go to Station 2", "Yaw 180").
- system_commands.jsonl: Technical commands for vehicle/suit configuration (e.g., "Switch to Omni C", "Open valve").
- verification_checks.jsonl: Status checks and readouts (e.g., "Check O2 pressure", "Confirm latch locked").
- observation_reports.jsonl: Scientific observations (e.g., "I see a vesicular rock", "Looks like basalt").
- coordination_events.jsonl: Workflow management (e.g., "Mark", "Stand by", "Copy").
- chatter_logs.jsonl: General conversation and non-actionable dialogue.

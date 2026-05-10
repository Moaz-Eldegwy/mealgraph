# Nutrition MAS Usage Guide

This guide explains how to use the Nutrition Multi-Agent System (MAS) by importing the `nutritionmas` module and calling a few simple functions. The system handles all the complex setup internally, so you only need to provide a list of API keys and optionally override model configurations.

## Steps to Use the System

### 1. Import the Module
Start by importing the `nutritionmas` module in your Python script:

```python
import nutritionmas
```

### 2. Define API Keys
Provide a list of API keys for the Gemini models. The system will cycle through these keys to distribute load and respect rate limits automatically (if enabled):

```python
api_keys = [
    "your_api_key1",
    "your_api_key2",
    # Add more keys as needed
]
```

### 3. (Optional) Override Model Configurations
If you want to customize the `model_name` or `params` for any model, define a `model_overrides` dictionary. Only these two fields can be overridden; other configurations are fixed in the module:

```python
model_overrides = {
    "main": {"model_name": "gemini-2.5-pro", "params": {"temperature": 0.5}},
    "agents_llm": {"model_name": "gemini-2.5-flash", "params": {"max_tokens": 6000}}
    # Add overrides for other models as needed
}
```

If you don’t need to override anything, you can skip this step or pass `None`.

### 4. (Optional) Enable/Disable Rate Limiting
By default, rate limiting is enabled using the pooled API keys. To disable it (e.g., for testing or when limits are not a concern), set `enable_rate_limiting=False`:

```python
nutritionmas.create_llm_instances(api_keys, model_overrides, enable_rate_limiting=False)
```

When disabled, the system simply cycles through keys without enforcing RPM/RPD waits or daily caps.

### 5. (Optional) Enable Debug Mode
By default, the system runs in **user mode**, displaying high-level progress messages for each agent and tool action, such as:

```
Coach Agent: Calling MedicalAssessmentAgent with task 'Assess user health'
Medical Assessment Agent: Using ComputationTool for 'Calculate BMI'
Coach Agent: Composing final response
```

To enable **debug mode** for detailed logging (e.g., raw LLM inputs/outputs), call the `debug` function before running the system:

```python
nutritionmas.debug(level='full', scopes={'agents': ['CoachAgent'], 'tools': ['all']})
```

- **Debug Levels**:
  - `full`: Shows raw LLM inputs and outputs (default for debug mode).
  - `output`: Shows only LLM outputs.
- **Debug Scopes** (optional):
  - `agents: ['all']` or `agents: ['CoachAgent', 'PlannerAgent']` to specify agents.
  - `tools: ['all']` or `tools: ['ComputationTool', 'WebSearchTool']` to specify tools.
  - If `scopes` is not provided, defaults to all agents and tools.

Example:

```python
nutritionmas.debug(level='output', scopes={'agents': ['CoachAgent']})
```

To run in user mode (default), simply skip the `debug` call.

### 6. System Logging
You can save the input and output of all agents and tools:

```python
nutritionmas.logging("Your Folder Path")
```

### 7. Initialize the System
Call the following functions in sequence to set up the system. These handle creating LLM instances, initializing tools and agents, and setting up the workflow:

```python
nutritionmas.create_llm_instances(api_keys, model_overrides, enable_rate_limiting=True)
nutritionmas.initialize_tools()
nutritionmas.initialize_agents()
nutritionmas.setup_workflow()
```

### 8. Run the System
Run the system in either interactive or simulation mode by calling the `run` function:

- **Interactive Mode**: Collects user data and allows real-time interaction.
- **Simulation Mode**: Uses predefined data and simulates user interactions (requires `simulated_users` argument).

```python
nutritionmas.run(simulate=False)  # For interactive mode
# or
nutritionmas.run(simulate=True, simulated_users=[...])   # For simulation mode
```

## Example Usage

Here’s a complete example with rate limiting enabled and user mode (default):

```python
import nutritionmas

# Define API keys as a list
api_keys = [
    "your_api_key1",
    "your_api_key2"
]

# Optional: Override model configurations
model_overrides = {
    "main": {"model_name": "gemini-2.5-pro", "params": {"temperature": 0.5}}
}

# Optional: Set logging directory
nutritionmas.logging("Your Folder Path")

# Initialize the system with rate limiting enabled
nutritionmas.create_llm_instances(api_keys, model_overrides, enable_rate_limiting=True)
nutritionmas.initialize_tools()
nutritionmas.initialize_agents()
nutritionmas.setup_workflow()

# Run in interactive mode
nutritionmas.run(simulate=False)
```

To enable debug mode with specific settings:

```python
import nutritionmas

api_keys = ["your_api_key1", "your_api_key2"]
model_overrides = {
    "main": {"model_name": "gemini-2.5-pro", "params": {"temperature": 0.5}}
}

nutritionmas.logging("Your Folder Path")
nutritionmas.debug(level='full', scopes={'agents': ['CoachAgent', 'PlannerAgent'], 'tools': ['ComputationTool']})
nutritionmas.create_llm_instances(api_keys, model_overrides, enable_rate_limiting=True)
nutritionmas.initialize_tools()
nutritionmas.initialize_agents()
nutritionmas.setup_workflow()
nutritionmas.run(simulate=False)
```

To disable rate limiting:

```python
nutritionmas.create_llm_instances(api_keys, model_overrides, enable_rate_limiting=False)
```

## Notes
- **User Mode Output**: In user mode (default, no `debug` call), the system shows high-level progress like:
  ```
  Coach Agent: Calling MedicalAssessmentAgent with task 'Assess user health'
  Medical Assessment Agent: Using ComputationTool for 'Calculate BMI'
  Planner Agent: Using WebSearchTool for 'Price of ingredients'
  Validation Agent: Completing validation
  Coach Agent: Composing final response
  ```
- **Debug Mode Output**: When `debug` is called, detailed logs (inputs/outputs or outputs only) are shown for specified agents/tools, based on the `level` and `scopes`.
- **API Key Pooling and Rate Limiting**: The system cycles through the provided API keys. When `enable_rate_limiting=True` (default), it enforces RPM (requests per minute) and RPD (requests per day) based on model-specific limits (e.g., 10 RPM / 250 RPD for gemini-2.5-flash). Keys exceeding daily limits are automatically dropped from the pool. Waits are calculated after response completion to exclude generation time. If a key is unavailable, it retries after 30 seconds.
- **Disabling Rate Limiting**: Set `enable_rate_limiting=False` to bypass all checks and waits, useful for local testing or unlimited scenarios. The system still cycles keys for load distribution.
- **Observed Waits in First Minute**: Small initial waits (e.g., 0.5s) may occur if there are rapid sequential calls or minor timing discrepancies during startup/system initialization. The code ensures no wait for truly first calls per key (empty timestamps), but variable generation times (~4-8s) combined with the 6s interval for flash can lead to partial waits if calls follow closely. This is by design to approximate rate limits conservatively without parallel tracking.
- **Interactive Mode**: You’ll be prompted to enter user information (e.g., name, age, dietary preferences) and can then ask nutrition-related questions. Type `exit` to quit.
- **Simulation Mode**: Provide a list of `simulated_users` (each with `user_profile`, `medical_history`, and `questions`). It processes each user's questions sequentially.
- **Error Handling**: Ensure at least one API key is provided, or you’ll get a `ValueError`. Each initialization step checks that the previous step has been completed.

That’s it! You can now use the Nutrition MAS system with minimal setup.

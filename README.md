# Constitutional AI for Education

This repository provides a comprehensive tutorial of how to apply Constitutional AI to fine-tune an LLM specifically for student support using efficient fine-tuning techniques.

## 🎓 Educational Principles

Our framework ensures AI responses:
- Foster critical thinking and independent problem-solving
- Provide appropriate scaffolding rather than direct answers
- Encourage deeper understanding of concepts
- Connect topics to broader learning contexts
- Maintain high academic standards

## 📚 Repository Structure

### `data/constitution_education.json`
Our educational constitution file defines the principles and guidelines for AI responses. It contains pairs of:
- **Critic Prompts**: Questions that evaluate responses based on educational principles
- **Revision Prompts**: Instructions for improving responses to better align with these principles

Examples of principles include:
- Encouraging reflection on problem-solving approaches
- Avoiding overly directive responses
- Promoting independent thinking
- Building connections to real-world examples
- Fostering curiosity and deeper exploration
- Addressing potential misconceptions
- Supporting student confidence and perseverance

This constitution serves as the foundation for generating our training dataset and guides the model toward becoming an effective teaching assistant.

### Notebooks

#### 1. `generate_dataset.ipynb`
- Creates training data using our educational constitution through a three-stage process:
  1. Generate initial responses to student queries
  2. Critique responses using our constitutional principles
  3. Revise responses based on the critique
- Demonstrates how to systematically apply educational principles to AI responses
- Shows how to structure and prepare data for fine-tuning

### 2. `run_sft.ipynb`
- Implements Supervised Fine-Tuning (SFT), the first training stage where:
  - The model learns to generate improved educational responses
  - Training uses the revised responses from our data generation process
  - The model begins to align with our educational principles
- Uses QLoRA for memory-efficient training, enabling:
  - Training on consumer GPUs
  - Preservation of base model knowledge while adding educational expertise
- Shows how to adapt a base model for educational tasks through:
  - Proper configuration of model parameters
  - Management of training data
  - Monitoring of learning progress
- Serves as the foundation for further refinement in the DPO stage

### 3. `run_dpo.ipynb`
- Implements Direct Preference Optimization (DPO), a second training stage that:
  - Teaches the model to prefer better teaching responses over less effective ones
  - Uses pairs of responses (better vs. worse) to learn from comparisons
  - Reinforces the educational principles established in our constitution
- Further refines the model by learning from preferences:
  - Original responses serve as examples of what to avoid
  - Revised responses show what good teaching looks like
  - The model learns to distinguish effective from ineffective teaching approaches
- Demonstrates how to maintain model stability through:
  - Careful management of reference and training models
  - Appropriate learning rate and training parameters
  - Balance between improvement and consistency
- Uses memory-efficient techniques to make training accessible on standard hardware

## 🛠 Technical Features

- Uses the Llama 3.2 (1B) model as the pretrained model to be fine-tuned
- Implements QLoRA for efficient fine-tuning
- Employs 4-bit quantization for reduced memory usage
- Includes optimizations for consumer GPU training
- Demonstrates proper adapter management for stable training
- Includes customizable educational constitution for defining teaching principles
- Demonstrates systematic application of educational guidelines

## 🚀 Getting Started

1. **Customize Educational Principles** (Optional)
- Review `data/constitution_education.json`
- Modify or add constitutional principles to match your educational goals
- Adjust critic and revision prompts as needed

2. **Generate Training Data**
- Run `generate_dataset.ipynb` to create your training dataset

3. **Initial Supervised Fine-tuning (SFT)**
- Run `run_sft.ipynb` to perform supervised fine-tuning

4. **Preference Learning using DPO**
- Run `run_dpo.ipynb` to refine the model using preferences

## 💻 Hardware Requirements

- Minimum: Single GPU with 8GB VRAM

## 📝 Notes

- This is a tutorial framework, not a production-ready system
- Meant for educational and research purposes
- Can be adapted for different educational contexts

## 💡 Customizing the Educational Constitution

The `constitution_education.json` file is designed to be easily modified for different educational contexts. You can:
- Add new educational principles
- Modify existing critique guidelines
- Adjust revision strategies

Example constitution entry:
```json
{
  "critic": "Does the response inspire the student to think about how they approached the problem?",
  "revision": "Rewrite the response to include thoughtful questions that encourage the student to reflect on their reasoning and approach."
}
```

This flexible structure allows educators to:
- Align AI responses with specific teaching methodologies
- Address particular subject matter needs
- Support different learning styles
- Focus on desired learning outcomes

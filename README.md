# Constitutional AI for Education: A Tutorial Framework

This repository provides a comprehensive tutorial framework for developing AI teaching assistants using Constitutional AI principles. It demonstrates how to create, fine-tune, and improve language models specifically for educational support using modern techniques like QLoRA and DPO.

## 🎯 Purpose

This repository serves as a proof-of-concept and practical guide for:
- Implementing Constitutional AI principles in educational contexts
- Creating AI systems that promote effective learning practices
- Demonstrating efficient fine-tuning techniques on consumer hardware
- Building a pipeline for continuous model improvement

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
- Implements Supervised Fine-Tuning (SFT)
- Uses QLoRA for memory-efficient training
- Shows how to adapt a base model for educational tasks
- Includes detailed explanations of configuration choices

### 3. `run_dpo.ipynb`
- Implements Direct Preference Optimization (DPO)
- Further refines the model using preference learning
- Demonstrates how to maintain model stability during training
- Includes memory-efficient training techniques

## 🛠 Technical Features

- Uses the Llama 3.2 (1B) model as base
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
- Modify critique prompts to align with your educational goals

3. **Initial Fine-tuning**
- Run `run_sft.ipynb` to perform supervised fine-tuning
- Adjust hyperparameters based on your hardware constraints

4. **Preference Learning**
- Run `run_dpo.ipynb` to refine the model using preferences
- Monitor training to ensure stable improvement

## 💻 Hardware Requirements

- Minimum: Single GPU with 8GB VRAM

## 📝 Notes

- This is a tutorial framework, not a production-ready system
- Meant for educational and research purposes
- Can be adapted for different educational contexts
- Demonstrates best practices in model fine-tuning

## 💡 Customizing the Educational Constitution

The `constitution_education.json` file is designed to be easily modified for different educational contexts. You can:
- Add new educational principles
- Modify existing critique guidelines
- Adjust revision strategies
- Focus on specific pedagogical approaches

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

#!/usr/bin/env python3
"""
Pipeline Orchestrator - Coordinates all 3 stages of analysis.

Stage 1: Guardrail Erosion (per turn)
Stage 2: RHO Calculation (per conversation)
Stage 3: PHI Aggregation (across conversations)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the 3-stage Vector Precognition pipeline.

    Stage 1: Process conversation turns → Guardrail metrics
    Stage 2: Calculate RHO → Robustness Index
    Stage 3: Aggregate RHO values → PHI Score
    """

    def __init__(
        self,
        vector_processor,
        robustness_calculator,
        fragility_calculator
    ):
        """
        Initialize orchestrator with components from Apps 1, 2, 3.

        Args:
            vector_processor: VectorPrecognitionProcessor (from App 1)
            robustness_calculator: RobustnessCalculator (from App 2)
            fragility_calculator: FragilityCalculator (from App 3)
        """
        self.vector_processor = vector_processor
        self.robustness_calculator = robustness_calculator
        self.fragility_calculator = fragility_calculator

        # Storage for conversations
        self.conversations = {}  # {conversation_id: conversation_data}
        self.current_conversation_id = None
        self.conversation_counter = 0

    def start_new_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Start a new conversation.

        Args:
            conversation_id: Optional custom ID

        Returns:
            Conversation ID
        """
        if conversation_id is None:
            self.conversation_counter += 1
            conversation_id = f"conversation_{self.conversation_counter}"

        self.current_conversation_id = conversation_id

        self.conversations[conversation_id] = {
            'id': conversation_id,
            'started_at': datetime.now(),
            'turns': [],
            'user_messages': [],
            'model_messages': [],
            'user_vectors': [],
            'model_vectors': [],
            'stage1_metrics': None,  # Guardrail metrics DataFrame
            'stage2_result': None,   # RHO calculation result
            'stage3_included': False,  # Whether included in PHI calculation
            'status': 'active'
        }

        logger.info(f"Started new conversation: {conversation_id}")
        return conversation_id

    def add_turn(
        self,
        user_message: str,
        model_message: str,
        user_vector: np.ndarray,
        model_vector: np.ndarray
    ):
        """
        Add a conversation turn (Stage 1: Real-time processing).

        Args:
            user_message: User's message text
            model_message: Model's response text
            user_vector: User message 2D vector
            model_vector: Model response 2D vector
        """
        if self.current_conversation_id is None:
            raise ValueError("No active conversation. Call start_new_conversation() first.")

        conv = self.conversations[self.current_conversation_id]

        # Store messages
        conv['user_messages'].append(user_message)
        conv['model_messages'].append(model_message)
        conv['user_vectors'].append(user_vector)
        conv['model_vectors'].append(model_vector)

        # Process turn through VectorPrecognition (Stage 1)
        self.vector_processor.process_turn(model_vector, user_vector)

        # Get current metrics
        turn_number = len(conv['user_messages'])
        conv['turns'].append({
            'turn': turn_number,
            'user_message': user_message,
            'model_message': model_message,
            'timestamp': datetime.now()
        })

        logger.info(f"Added turn {turn_number} to {self.current_conversation_id}")

    def get_stage1_metrics(self) -> pd.DataFrame:
        """
        Get Stage 1 metrics (Guardrail Erosion) for current conversation.

        Returns:
            Metrics DataFrame
        """
        if self.current_conversation_id is None:
            return pd.DataFrame()

        metrics_df = self.vector_processor.get_metrics()
        self.conversations[self.current_conversation_id]['stage1_metrics'] = metrics_df

        return metrics_df

    def calculate_stage2_rho(self) -> Dict:
        """
        Calculate Stage 2 (RHO) for current conversation.

        Returns:
            RHO calculation results
        """
        if self.current_conversation_id is None:
            raise ValueError("No active conversation")

        conv = self.conversations[self.current_conversation_id]

        # Get Stage 1 metrics
        if conv['stage1_metrics'] is None:
            metrics_df = self.get_stage1_metrics()
        else:
            metrics_df = conv['stage1_metrics']

        # Calculate RHO using Stage 2 calculator
        rho_result = self.robustness_calculator.analyze_conversation(
            metrics_df,
            conversation_id=self.current_conversation_id
        )

        conv['stage2_result'] = rho_result
        conv['status'] = 'analyzed'

        logger.info(f"Calculated RHO for {self.current_conversation_id}: {rho_result['final_rho']:.3f}")

        return rho_result

    def end_conversation(self) -> str:
        """
        End current conversation and calculate final metrics.

        Returns:
            Conversation ID that was ended
        """
        if self.current_conversation_id is None:
            raise ValueError("No active conversation to end")

        conv_id = self.current_conversation_id
        conv = self.conversations[conv_id]

        # Calculate Stage 2 if not already done
        if conv['stage2_result'] is None:
            self.calculate_stage2_rho()

        conv['ended_at'] = datetime.now()
        conv['status'] = 'completed'

        logger.info(f"Ended conversation: {conv_id}")

        # Reset current conversation
        self.current_conversation_id = None

        return conv_id

    def calculate_stage3_phi(self, model_name: str = "Current Model") -> Dict:
        """
        Calculate Stage 3 (PHI) across all completed conversations.

        Args:
            model_name: Name of the model being evaluated

        Returns:
            PHI calculation results
        """
        # Get RHO values from all completed conversations
        rho_values = []
        test_ids = []

        for conv_id, conv in self.conversations.items():
            if conv['stage2_result'] is not None:
                rho_values.append(conv['stage2_result']['final_rho'])
                test_ids.append(conv_id)
                conv['stage3_included'] = True

        if not rho_values:
            logger.warning("No conversations with RHO values for PHI calculation")
            return {'phi_score': 0.0, 'classification': 'N/A'}

        # Calculate PHI using Stage 3 calculator
        phi_result = self.fragility_calculator.evaluate_model(
            model_name,
            rho_values,
            test_ids
        )

        logger.info(f"Calculated PHI for {model_name}: {phi_result['phi_score']:.4f}")

        return phi_result

    def get_current_status(self) -> Dict:
        """
        Get current pipeline status.

        Returns:
            Status dictionary with all stages
        """
        if self.current_conversation_id is None:
            return {
                'has_active_conversation': False,
                'total_conversations': len(self.conversations),
                'completed_conversations': sum(1 for c in self.conversations.values() if c['status'] == 'completed')
            }

        conv = self.conversations[self.current_conversation_id]
        metrics_df = self.get_stage1_metrics()

        status = {
            'has_active_conversation': True,
            'conversation_id': self.current_conversation_id,
            'total_turns': len(conv['user_messages']),
            'total_conversations': len(self.conversations),
            'completed_conversations': sum(1 for c in self.conversations.values() if c['status'] == 'completed'),
            'stage1': {
                'peak_risk': metrics_df['RiskSeverity_Model'].max() if len(metrics_df) > 0 else 0.0,
                'peak_likelihood': metrics_df['Likelihood_L(N)'].max() if len(metrics_df) > 0 else 0.0,
                'current_risk': metrics_df['RiskSeverity_Model'].iloc[-1] if len(metrics_df) > 0 else 0.0
            }
        }

        if conv['stage2_result'] is not None:
            status['stage2'] = {
                'final_rho': conv['stage2_result']['final_rho'],
                'classification': conv['stage2_result']['classification'],
                'is_robust': conv['stage2_result']['is_robust']
            }

        return status

    def get_conversation_history(self) -> List[Dict]:
        """
        Get history of all conversations.

        Returns:
            List of conversation summaries
        """
        history = []

        for conv_id, conv in self.conversations.items():
            summary = {
                'id': conv_id,
                'turns': len(conv['user_messages']),
                'status': conv['status'],
                'started_at': conv['started_at'].strftime('%Y-%m-%d %H:%M:%S'),
            }

            if 'ended_at' in conv:
                summary['ended_at'] = conv['ended_at'].strftime('%Y-%m-%d %H:%M:%S')

            if conv['stage2_result'] is not None:
                summary['rho'] = conv['stage2_result']['final_rho']
                summary['classification'] = conv['stage2_result']['classification']

            history.append(summary)

        return history

    def reset(self):
        """Reset the entire pipeline (for new session)."""
        self.conversations = {}
        self.current_conversation_id = None
        self.conversation_counter = 0
        self.vector_processor.reset()
        self.robustness_calculator.clear()
        self.fragility_calculator.clear()

        logger.info("Pipeline reset")

    def export_session(self) -> Dict:
        """
        Export entire session data.

        Returns:
            Dictionary with all session data
        """
        return {
            'total_conversations': len(self.conversations),
            'completed_conversations': sum(1 for c in self.conversations.values() if c['status'] == 'completed'),
            'conversations': {
                conv_id: {
                    'id': conv['id'],
                    'turns': conv['turns'],
                    'status': conv['status'],
                    'rho': conv['stage2_result']['final_rho'] if conv['stage2_result'] else None,
                    'classification': conv['stage2_result']['classification'] if conv['stage2_result'] else None
                }
                for conv_id, conv in self.conversations.items()
            },
            'export_timestamp': datetime.now().isoformat()
        }

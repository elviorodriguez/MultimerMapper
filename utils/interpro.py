#!/usr/bin/env python3
"""
InterPro Domain Search Script with Segment Analysis
Searches for protein domains in a given sequence using the InterPro REST API
and analyzes domain coverage across specified segments.

Usage: python interpro_search.py <protein_sequence> [segment_starts] [options]
"""

import os
import requests
import time
import json
import argparse
import csv
from typing import Dict, List, Optional, Tuple
import re

from utils.logger_setup import configure_logger

class DomainSummarizer:
    """Handles domain name summarization using pure word enrichment from InterPro data."""
    
    def __init__(self):
        # Generic stop words only (no domain-specific terms)
        self.stop_words = {
            'and', 'or', 'with', 'of', 'the', 'a', 'an', 'in', 'at', 'to', 'for',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may',
            'might', 'must', 'shall', 'this', 'that', 'these', 'those'
        }

    def normalize_text(self, text: str) -> List[str]:
        """Clean and tokenize text, removing only generic stop words."""
        if not text or text is None:
            return []
        
        # Convert to lowercase and clean
        text = str(text).lower()
        # Replace separators with spaces but keep meaningful punctuation context
        text = re.sub(r'[_\-/]', ' ', text)
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove parentheses content only if it's very generic
        text = re.sub(r'\([^)]*\)', '', text)
        
        words = text.split()
        
        # Filter out only generic stop words and very short words
        cleaned_words = []
        for word in words:
            if (word not in self.stop_words and 
                len(word) > 1 and 
                not word.isdigit() and
                word.isalpha()):
                cleaned_words.append(word)
        
        return cleaned_words
    
    def extract_word_ngrams(self, words: List[str], max_n: int = 4) -> List[str]:
        """Extract n-grams (word combinations) from words, up to 4-grams."""
        if not words:
            return []
        
        ngrams = []
        
        # Add individual words (1-grams)
        ngrams.extend(words)
        
        # Add n-grams for n=2 to max_n
        for n in range(2, min(max_n + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.append(ngram)
        
        return ngrams
    
    def calculate_information_content(self, ngram: str, all_ngrams: List[str]) -> float:
        """Calculate information content based on frequency and specificity."""
        words_in_ngram = len(ngram.split())
        frequency = all_ngrams.count(ngram)
        total_ngrams = len(all_ngrams)
        
        if frequency == 0:
            return 0.0
        
        # Base frequency score
        freq_score = frequency / total_ngrams
        
        # Specificity bonus: longer phrases are more specific
        specificity_score = words_in_ngram * 0.15
        
        # Rarity bonus: moderately rare terms are often more informative than very common ones
        # But not so rare that they're noise
        if 0.1 <= freq_score <= 0.7:
            rarity_bonus = 0.1
        else:
            rarity_bonus = 0.0
        
        return freq_score + specificity_score + rarity_bonus
    
    def analyze_domain_patterns(self, domains: List[Dict]) -> Dict:
        """Analyze word patterns purely from the available InterPro data."""
        if not domains:
            return {'summary': 'No domain detected', 'confidence': 0.0}
        
        # Extract all n-grams from domain information
        all_ngrams = []
        domain_sources = []  # Track which domains contribute which n-grams
        
        for i, domain in enumerate(domains):
            name = domain.get('name') or ''
            description = domain.get('description') or ''
            interpro_name = domain.get('interpro_name') or ''
            
            # Process all available text sources
            text_sources = [name, description, interpro_name]
            domain_ngrams = []
            
            for text in text_sources:
                if text and text.lower() not in ['none', 'unknown', 'no description']:
                    words = self.normalize_text(text)
                    ngrams = self.extract_word_ngrams(words)
                    domain_ngrams.extend(ngrams)
                    all_ngrams.extend(ngrams)
            
            domain_sources.append({
                'domain_idx': i,
                'ngrams': domain_ngrams,
                'name': name,
                'database': domain.get('database', 'Unknown')
            })
        
        if not all_ngrams:
            return {'summary': 'Unknown domain', 'confidence': 0.0}
        
        # Calculate information content for each unique n-gram
        unique_ngrams = list(set(all_ngrams))
        ngram_scores = {}
        
        for ngram in unique_ngrams:
            info_content = self.calculate_information_content(ngram, all_ngrams)
            ngram_scores[ngram] = info_content
        
        # Sort by information content
        top_ngrams = sorted(ngram_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build summary from most informative patterns
        summary = self.build_data_driven_summary(top_ngrams, len(domains), domain_sources)
        confidence = top_ngrams[0][1] if top_ngrams else 0.0
        
        return {
            'summary': summary,
            'confidence': confidence,
            'top_patterns': top_ngrams[:5],
            'domain_count': len(domains)
        }
    
    def build_data_driven_summary(self, top_ngrams: List[Tuple[str, float]], 
                                 domain_count: int, domain_sources: List[Dict]) -> str:
        """Build summary purely from the most informative n-grams in the data."""
        if not top_ngrams:
            return "Unknown domain"
        
        # Get the most informative n-gram
        best_ngram, best_score = top_ngrams[0]
        
        # Capitalize appropriately (first letter of each word)
        primary_terms = []
        for word in best_ngram.split():
            # Handle common abbreviations that should stay uppercase
            if len(word) <= 3 and word.upper() in ['DNA', 'RNA', 'ATP', 'GTP', 'SH2', 'SH3']:
                primary_terms.append(word.upper())
            else:
                primary_terms.append(word.capitalize())
        
        primary_summary = ' '.join(primary_terms)
        
        # For multiple domains, try to add complementary information
        if domain_count > 1 and len(top_ngrams) > 1:
            # Look for a complementary n-gram that doesn't overlap too much
            best_words = set(best_ngram.split())
            
            for ngram, score in top_ngrams[1:4]:  # Check next few top ngrams
                ngram_words = set(ngram.split())
                
                # If this adds new information and has decent score
                if (len(ngram_words - best_words) > 0 and 
                    score > best_score * 0.3):  # At least 30% as informative
                    
                    # Add the new information
                    new_words = ngram_words - best_words
                    if new_words:
                        additional = ' '.join(sorted(new_words))
                        additional = ' '.join([w.capitalize() for w in additional.split()])
                        primary_summary = f"{primary_summary}/{additional}"
                        break
            
            # Add count for many domains
            if domain_count > 3:
                primary_summary += f" (+{domain_count-1} domains)"
            elif domain_count > 1:
                primary_summary += " (multiple)"
        
        return primary_summary

    def summarize_domains(self, domains: List[Dict]) -> str:
        """Summarize domains using word enrichment analysis."""
        if not domains:
            return "No domain detected"
        
        # Use the new pattern analysis approach
        analysis_result = self.analyze_domain_patterns(domains)
        return analysis_result['summary']

class InterProSearcher:
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/interpro/api"
        self.headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'InterProDomainSearcher/1.0'
        }
        self.summarizer = DomainSummarizer()
    
    def validate_sequence(self, sequence: str) -> bool:
        """Validate that the sequence contains only valid amino acid characters."""
        valid_aa = set('ACDEFGHIKLMNPQRSTVWYXU')
        return all(aa.upper() in valid_aa for aa in sequence.strip())
    
    def parse_segments(self, segment_string: str, sequence_length: int) -> List[Tuple[int, int]]:
        """Parse segment start positions and calculate end positions."""
        try:
            starts = [int(x.strip()) for x in segment_string.split(',')]
            segments = []
            
            for i, start in enumerate(starts):
                if i < len(starts) - 1:
                    end = starts[i + 1] - 1
                else:
                    end = sequence_length
                segments.append((start, end))
            
            return segments
        except ValueError:
            raise ValueError("Invalid segment format. Use comma-separated integers.")
    
    def submit_search(self, sequence: str) -> Optional[str]:
        """Submit a sequence search to InterPro and return job ID."""
        interproscan_url = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5/run"
        
        data = {
            'email': 'user@example.com',
            'sequence': sequence.strip(),
            'goterms': 'false',
            'pathways': 'false'
        }
        
        try:
            response = requests.post(interproscan_url, data=data, headers=self.headers)
            if response.status_code == 200:
                job_id = response.text.strip()
                print(f"Job submitted successfully. Job ID: {job_id}")
                return job_id
            else:
                print(f"Error submitting job: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return None
    
    def check_job_status(self, job_id: str) -> str:
        """Check the status of a submitted job."""
        status_url = f"https://www.ebi.ac.uk/Tools/services/rest/iprscan5/status/{job_id}"
        
        try:
            response = requests.get(status_url)
            if response.status_code == 200:
                return response.text.strip()
            else:
                return "ERROR"
        except requests.RequestException:
            return "ERROR"
    
    def get_results(self, job_id: str) -> Optional[Dict]:
        """Retrieve results for a completed job."""
        result_url = f"https://www.ebi.ac.uk/Tools/services/rest/iprscan5/result/{job_id}/json"
        
        try:
            response = requests.get(result_url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error retrieving results: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
    
    def wait_for_completion(self, job_id: str, max_wait: int = 300) -> bool:
        """Wait for job completion with timeout."""
        print("Waiting for job completion...", end="")
        wait_time = 0
        
        while wait_time < max_wait:
            status = self.check_job_status(job_id)
            print(".", end="", flush=True)
            
            if status == "FINISHED":
                print("\nJob completed!")
                return True
            elif status in ["ERROR", "FAILURE"]:
                print(f"\nJob failed with status: {status}")
                return False
            
            time.sleep(5)
            wait_time += 5
        
        print(f"\nTimeout after {max_wait} seconds")
        return False
    
    def extract_domain_info(self, results: Dict) -> List[Dict]:
        """Extract domain information from InterPro results."""
        domains = []
        
        if not results or 'results' not in results:
            return domains
        
        for result in results['results']:
            matches = result.get('matches', [])
            
            for match in matches:
                signature = match.get('signature', {})
                locations = match.get('locations', [])
                
                for location in locations:
                    # Handle None values safely
                    accession = signature.get('accession')
                    name = signature.get('name')
                    description = signature.get('description')
                    
                    domain_info = {
                        'accession': str(accession) if accession else 'Unknown',
                        'name': str(name) if name else 'Unknown',
                        'description': str(description) if description else 'No description',
                        'database': signature.get('signatureLibraryRelease', {}).get('library', 'Unknown'),
                        'start': location.get('start', 0),
                        'end': location.get('end', 0),
                        'score': location.get('score', 'N/A')
                    }
                    
                    # Add InterPro entry info if available
                    interpro_entry = signature.get('entry')
                    if interpro_entry:
                        domain_info['interpro_accession'] = interpro_entry.get('accession', '')
                        domain_info['interpro_name'] = interpro_entry.get('name', '')
                        domain_info['interpro_type'] = interpro_entry.get('type', '')
                    
                    domains.append(domain_info)
        
        return domains
    
    def analyze_segment_domains(self, domains: List[Dict], segments: List[Tuple[int, int]]) -> List[Dict]:
        """Analyze which domains overlap with each segment."""
        segment_analysis = []
        
        for i, (start, end) in enumerate(segments):
            segment_domains = []
            partial_domains = []
            
            for domain in domains:
                domain_start = domain['start']
                domain_end = domain['end']
                
                # Check if domain overlaps with segment
                if domain_end >= start and domain_start <= end:
                    # Check if domain is fully contained
                    if domain_start >= start and domain_end <= end:
                        segment_domains.append(domain)
                    else:
                        # Domain is partially contained
                        partial_domains.append(domain)
                        segment_domains.append(domain)  # Include in analysis but mark as partial
            
            # Summarize domains for this segment
            domain_summary = self.summarizer.summarize_domains(segment_domains)
            
            # Add partial indication if needed
            if partial_domains:
                if domain_summary != "No domain detected":
                    domain_summary += " (partial)"
            
            segment_analysis.append({
                'segment': i + 1,
                'start': start,
                'end': end,
                'domain': domain_summary,
                'num_domains': len(segment_domains),
                'partial_domains': len(partial_domains) > 0
            })
        
        return segment_analysis
    
    def format_results(self, results: Dict, output_file: Optional[str] = None) -> None:
        """Format and display the search results."""
        output_lines = []
        
        if not results or 'results' not in results:
            output_lines.append("No results found or invalid response format.")
            self._write_output(output_lines, output_file)
            return
        
        for result in results['results']:
            sequence_id = result.get('xref', [{}])[0].get('id', 'Unknown')
            output_lines.append(f"\n=== Results for sequence: {sequence_id} ===")
            output_lines.append(f"Sequence length: {result.get('length', 'Unknown')} amino acids")
            
            matches = result.get('matches', [])
            if not matches:
                output_lines.append("No domain matches found.")
                continue
            
            output_lines.append(f"\nFound {len(matches)} domain matches:")
            output_lines.append("-" * 80)
            
            for i, match in enumerate(matches, 1):
                signature = match.get('signature', {})
                accession = signature.get('accession', 'Unknown')
                name = signature.get('name', 'Unknown')
                description = signature.get('description', 'No description')
                db_name = signature.get('signatureLibraryRelease', {}).get('library', 'Unknown')
                
                output_lines.append(f"{i}. {name} ({accession})")
                output_lines.append(f"   Database: {db_name}")
                output_lines.append(f"   Description: {description}")
                
                locations = match.get('locations', [])
                if locations:
                    output_lines.append("   Locations:")
                    for loc in locations:
                        start = loc.get('start', '?')
                        end = loc.get('end', '?')
                        score = loc.get('score', 'N/A')
                        output_lines.append(f"     - Position {start}-{end} (Score: {score})")
                
                interpro_entry = signature.get('entry')
                if interpro_entry:
                    entry_acc = interpro_entry.get('accession', '')
                    entry_name = interpro_entry.get('name', '')
                    entry_type = interpro_entry.get('type', '')
                    output_lines.append(f"   InterPro Entry: {entry_name} ({entry_acc}) - {entry_type}")
                
                output_lines.append("")
        
        self._write_output(output_lines, output_file)
    
    def save_segment_analysis(self, segment_analysis: List[Dict], filename: str) -> None:
        """Save segment analysis to a CSV file."""
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['segment', 'start', 'end', 'domain']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for segment in segment_analysis:
                writer.writerow({
                    'segment': segment['segment'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'domain': segment['domain']
                })
        
        print(f"Segment analysis saved to: {filename}")
    
    def display_segment_analysis(self, segment_analysis: List[Dict], output_file: Optional[str] = None) -> None:
        """Display segment analysis results."""
        output_lines = ["\n=== SEGMENT ANALYSIS ==="]
        output_lines.append(f"{'Segment':<8} {'Start':<6} {'End':<6} {'Domain'}")
        output_lines.append("-" * 60)
        
        for segment in segment_analysis:
            partial_indicator = " (P)" if segment['partial_domains'] else ""
            output_lines.append(f"{segment['segment']:<8} {segment['start']:<6} {segment['end']:<6} {segment['domain']}{partial_indicator}")
        
        output_lines.append("\nLegend: (P) = Partially contained domain")
        
        self._write_output(output_lines, output_file)
    
    def _write_output(self, lines: List[str], output_file: Optional[str] = None) -> None:
        """Write output to file and/or console."""
        output_text = '\n'.join(lines)
        print(output_text)
        
        if output_file:
            with open(output_file, 'a') as f:
                f.write(output_text + '\n')
    
    def search_domains(self, sequence: str, segments: Optional[List[Tuple[int, int]]] = None, 
                      output_file: Optional[str] = None, csv_file: Optional[str] = None) -> None:
        """Main method to search for domains in a protein sequence."""
        print(f"Searching domains for sequence of length {len(sequence)} amino acids...")
        
        if not self.validate_sequence(sequence):
            print("Error: Invalid protein sequence. Please use standard amino acid codes.")
            return
        
        # Clear output file if it exists
        if output_file:
            open(output_file, 'w').close()
        
        job_id = self.submit_search(sequence)
        if not job_id:
            return
        
        if not self.wait_for_completion(job_id):
            return
        
        results = self.get_results(job_id)
        if not results:
            print("Failed to retrieve results.")
            return
        
        # Display full results
        self.format_results(results, output_file)
        
        # Analyze segments if provided
        if segments:
            domains = self.extract_domain_info(results)
            segment_analysis = self.analyze_segment_domains(domains, segments)
            
            # Display segment analysis
            self.display_segment_analysis(segment_analysis, output_file)
            
            # Save to CSV if requested
            if csv_file:
                self.save_segment_analysis(segment_analysis, csv_file)


def submit_interpro_jobs(prot_IDs: list, prot_seqs: list, logger = None):
    """Submit InterPro jobs for all protein sequences and return job tracking dict."""
    if logger is None:
        logger = configure_logger()(__name__)
    
    searcher = InterProSearcher()
    interpro_jobs = {}
    
    logger.info("Submitting InterPro domain search jobs...")
    
    for i, (prot_id, sequence) in enumerate(zip(prot_IDs, prot_seqs)):
        logger.info(f"Submitting job for protein {i+1}/{len(prot_IDs)}: {prot_id}")
        
        job_id = searcher.submit_search(sequence)
        if job_id:
            interpro_jobs[prot_id] = {
                'job_id': job_id,
                'sequence': sequence,
                'status': 'RUNNING',
                'results': None,
                'summary': None
            }
            logger.info(f"  Job ID: {job_id}")
        else:
            logger.error(f"Failed to submit job for {prot_id}")
            interpro_jobs[prot_id] = {
                'job_id': None,
                'sequence': sequence,
                'status': 'FAILED',
                'results': None,
                'summary': 'Submission failed'
            }
    
    logger.info(f"Submitted {len([j for j in interpro_jobs.values() if j['job_id']])} InterPro jobs successfully")
    return interpro_jobs


def create_empty_interpro_results(sequence, reason):
    """Create an empty InterPro results structure with failure reason."""
    start = 1
    end = len(sequence)
    return {
    "interproscan-version": "5.75-106.0",
    "failure_reason": reason,
    "results": [{
            "sequence": sequence,
            "md5": "",
            "matches": [{
            "signature": {
                "accession": reason,
                "name": reason,
                "description": reason,
                "type": "REGION"
            },
            "locations": [
                {
                "start": start,
                "end": end
                }
            ]
            }],
            "xref": [
                {
                    "name": "FAILED_JOB",
                    "id": "FAILED_JOB"
                }
            ]
        }
    ]
    }


def check_and_process_interpro_results(interpro_jobs: dict, out_path: str, logger = None, max_wait_per_cycle: int = 300):
    """Check InterPro job status and process results with user interaction."""
    if logger is None:
        logger = configure_logger()(__name__)
    
    searcher = InterProSearcher()
    
    # Create interpro subfolder
    interpro_folder = os.path.join(out_path, "domains", "interpro")
    os.makedirs(interpro_folder, exist_ok=True)
    
    while True:
        # Check status of all jobs
        pending_jobs = []
        completed_jobs = []
        
        for prot_id, job_info in interpro_jobs.items():
            if job_info['status'] == 'RUNNING' and job_info['job_id']:
                status = searcher.check_job_status(job_info['job_id'])
                if status == 'FINISHED':
                    # Get results
                    results = searcher.get_results(job_info['job_id'])
                    if results:
                        domains = searcher.extract_domain_info(results)
                        summary = searcher.summarizer.summarize_domains(domains)
                        
                        job_info['status'] = 'COMPLETED'
                        job_info['results'] = results
                        job_info['summary'] = summary
                        
                        # Save results to file
                        result_file = os.path.join(interpro_folder, f"{prot_id}_interpro_results.json")
                        with open(result_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        
                        completed_jobs.append(prot_id)
                        logger.info(f"InterPro job completed for {prot_id}: {summary}")
                    else:
                        job_info['status'] = 'FAILED'
                        job_info['summary'] = 'Failed to retrieve results'
                        
                        # Generate empty results file
                        empty_results = create_empty_interpro_results(job_info.get('sequence', ''), 'Failed to retrieve results from InterPro server')
                        result_file = os.path.join(interpro_folder, f"{prot_id}_interpro_results.json")
                        with open(result_file, 'w') as f:
                            json.dump(empty_results, f, indent=2)
                        
                        logger.error(f"Failed to retrieve InterPro results for {prot_id}")
                elif status in ['ERROR', 'FAILURE']:
                    job_info['status'] = 'FAILED'
                    job_info['summary'] = f'Job failed: {status}'
                    
                    # Generate empty results file
                    empty_results = create_empty_interpro_results(job_info.get('sequence', ''), f'InterPro job failed with status: {status}')
                    result_file = os.path.join(interpro_folder, f"{prot_id}_interpro_results.json")
                    with open(result_file, 'w') as f:
                        json.dump(empty_results, f, indent=2)
                    
                    logger.error(f"InterPro job failed for {prot_id}: {status}")
                else:
                    pending_jobs.append(prot_id)
        
        # If all jobs are done, break
        if not pending_jobs:
            logger.info("All InterPro jobs completed!")
            break
        
        # Ask user if they want to wait
        logger.info(f"InterPro jobs status: {len(completed_jobs)} completed, {len(pending_jobs)} pending")
        logger.info(f"Pending jobs for: {', '.join(pending_jobs)}")
        
        user_input = input(f"Do you want to wait up to {max_wait_per_cycle} seconds for pending InterPro jobs? (y/n): ").lower().strip()
        
        if user_input == 'y':
            logger.info(f"Waiting up to {max_wait_per_cycle} seconds for jobs to complete...")
            start_time = time.time()
            
            while time.time() - start_time < max_wait_per_cycle and pending_jobs:
                time.sleep(10)  # Check every 10 seconds
                
                # Recheck pending jobs
                still_pending = []
                for prot_id in pending_jobs:
                    job_info = interpro_jobs[prot_id]
                    if job_info['job_id']:
                        status = searcher.check_job_status(job_info['job_id'])
                        if status == 'FINISHED':
                            results = searcher.get_results(job_info['job_id'])
                            if results:
                                domains = searcher.extract_domain_info(results)
                                summary = searcher.summarizer.summarize_domains(domains)
                                
                                job_info['status'] = 'COMPLETED'
                                job_info['results'] = results
                                job_info['summary'] = summary
                                
                                # Save results
                                result_file = os.path.join(interpro_folder, f"{prot_id}_interpro_results.json")
                                with open(result_file, 'w') as f:
                                    json.dump(results, f, indent=2)
                                
                                logger.info(f"InterPro job completed for {prot_id}: {summary}")
                            else:
                                job_info['status'] = 'FAILED'
                                job_info['summary'] = 'Failed to retrieve results'
                        elif status in ['ERROR', 'FAILURE']:
                            job_info['status'] = 'FAILED'
                            job_info['summary'] = f'Job failed: {status}'
                        else:
                            still_pending.append(prot_id)
                
                pending_jobs = still_pending
                
                if not pending_jobs:
                    logger.info("All jobs completed during wait!")
                    break
            
            if pending_jobs:
                # Generate empty files for jobs that timed out
                for prot_id in pending_jobs:
                    interpro_jobs[prot_id]['summary'] = f'Timeout after {max_wait_per_cycle} seconds'
                    interpro_jobs[prot_id]['status'] = 'TIMEOUT'
                    
                    # Generate empty results file
                    empty_results = create_empty_interpro_results(interpro_jobs[prot_id].get('sequence', ''), f'InterPro job timed out after {max_wait_per_cycle} seconds')
                    result_file = os.path.join(interpro_folder, f"{prot_id}_interpro_results.json")
                    with open(result_file, 'w') as f:
                        json.dump(empty_results, f, indent=2)
                
                logger.warning(f"Wait time exceeded. {len(pending_jobs)} jobs still pending.")
        else:
            # User chose not to wait, mark pending jobs as timeout and generate empty files
            for prot_id in pending_jobs:
                interpro_jobs[prot_id]['summary'] = 'Waiting time out'
                interpro_jobs[prot_id]['status'] = 'TIMEOUT'
                
                # Generate empty results file
                empty_results = create_empty_interpro_results(interpro_jobs[prot_id].get('sequence', ''), 'User chose not to wait for InterPro results')
                result_file = os.path.join(interpro_folder, f"{prot_id}_interpro_results.json")
                with open(result_file, 'w') as f:
                    json.dump(empty_results, f, indent=2)
                
            logger.info("User chose not to wait. Proceeding with available results.")
            break
    
    return interpro_jobs



def main():
    parser = argparse.ArgumentParser(
        description="Search for protein domains using InterPro REST API with segment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python interpro_search.py "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
  
  # With segments
  python interpro_search.py "MGSSHHHHHHSSGLVPRGSHMKIVRILERSKEP" "1,11"
  
  # With output files
  python interpro_search.py "MGSSHHHHHHSSGLVPRGSHMKIVRILERSKEP" "1,11" --output results.txt --csv segments.csv
        """
    )
    
    parser.add_argument('sequence', help='Protein sequence to search')
    parser.add_argument('segments', nargs='?', help='Comma-separated segment start positions (e.g., "1,11,25")')
    parser.add_argument('--output', '-o', help='Output file for detailed results')
    parser.add_argument('--csv', '-c', help='CSV file for segment analysis table')
    parser.add_argument('--max-wait', type=int, default=300, help='Maximum wait time in seconds')
    
    args = parser.parse_args()
    
    sequence = ''.join(args.sequence.split()).upper()
    
    if len(sequence) == 0:
        print("Error: Empty sequence provided")
        return
    
    segments = None
    if args.segments:
        try:
            searcher = InterProSearcher()
            segments = searcher.parse_segments(args.segments, len(sequence))
            print(f"Analyzing {len(segments)} segments:")
            for i, (start, end) in enumerate(segments, 1):
                print(f"  Segment {i}: {start}-{end}")
        except ValueError as e:
            print(f"Error parsing segments: {e}")
            return
    
    print(f"\nInterPro Domain Search")
    print(f"Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
    print(f"Length: {len(sequence)} amino acids")
    print("-" * 50)
    
    searcher = InterProSearcher()
    searcher.search_domains(sequence, segments, args.output, args.csv)

if __name__ == "__main__":
    main()
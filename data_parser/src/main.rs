use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn main() {
    println!("Starting arXiv data parsing...");

    let input_path = "../data/arxiv-metadata-oai-snapshot.json";
    if !Path::new(input_path).exists() {
        println!("Input file not found at {}. Please run download_data.py first.", input_path);
        // We don't want to panic here because we're just building the skeleton
        return;
    }

    let file = File::open(input_path).expect("Failed to open file");
    let reader = BufReader::new(file);

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut valid_nodes = HashSet::new();

    // Pass 1: Collect valid math/physics papers
    println!("Pass 1: Identifying math/physics papers...");
    for line_result in reader.lines() {
        if let Ok(line) = line_result {
            if let Ok(json) = serde_json::from_str::<Value>(&line) {
                let id = json["id"].as_str().unwrap_or("").to_string();
                let categories = json["categories"].as_str().unwrap_or("");
                
                if categories.contains("math") || categories.contains("physics") || categories.contains("astro-ph") || categories.contains("cond-mat") || categories.contains("gr-qc") || categories.contains("hep-") || categories.contains("nlin") || categories.contains("nucl-") || categories.contains("quant-ph") {
                    valid_nodes.insert(id.clone());
                    
                    let date = json["update_date"].as_str().unwrap_or("").to_string();
                    let abs = json["abstract"].as_str().unwrap_or("").replace("\n", " ");
                    
                    nodes.push((id.clone(), categories.to_string(), date, abs));
                }
            }
        }
    }
    
    println!("Found {} valid math/physics papers.", valid_nodes.len());

    // Pass 2: Re-read to extract citations if available
    // Kaggle metadata natively might not have citations, but if it's an augmented dataset:
    let file = File::open(input_path).expect("Failed to open file");
    let reader = BufReader::new(file);
    
    println!("Pass 2: Extracting citations...");
    for line_result in reader.lines() {
        if let Ok(line) = line_result {
            if let Ok(json) = serde_json::from_str::<Value>(&line) {
                let id = json["id"].as_str().unwrap_or("").to_string();
                
                if valid_nodes.contains(&id) {
                    if let Some(refs) = json["references"].as_array() {
                        for r in refs {
                            if let Some(r_str) = r.as_str() {
                                let target_id = r_str.to_string();
                                if valid_nodes.contains(&target_id) {
                                    edges.push((id.clone(), target_id));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("Found {} citation edges within the domain.", edges.len());

    // Write Nodes CSV
    let mut wtr = csv::Writer::from_path("../data/nodes.csv").unwrap();
    wtr.write_record(&["id", "categories", "date", "abstract"]).unwrap();
    for (id, cat, date, abs) in nodes {
        wtr.write_record(&[id, cat, date, abs]).unwrap();
    }
    wtr.flush().unwrap();
    println!("Wrote nodes.csv");

    // Write Edges CSV
    let mut wtr = csv::Writer::from_path("../data/edges.csv").unwrap();
    wtr.write_record(&["source", "target"]).unwrap();
    for (source, target) in edges {
        wtr.write_record(&[source, target]).unwrap();
    }
    wtr.flush().unwrap();
    println!("Wrote edges.csv");
    
    println!("Data parsing completed successfully!");
}

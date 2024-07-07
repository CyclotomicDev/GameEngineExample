//////////////////////////////////////////////////
// Conversion from map data to usable formats
//////////////////////////////////////////////////

use anyhow::{anyhow, Result};
use geo_types;
use geojson::GeoJson;
use serde::{de, Deserializer, Deserialize};
use serde_json;
use std::fs;

#[derive(Debug)]
struct Color(u8, u8, u8);

impl Default for Color {
    fn default() -> Self {
        Self(0, 0, 0)
    }
}

fn arbitrary_geometry<'de, D>(deserializer: D) -> Result<Vec<geo_types::Polygon<f64>>, D::Error>
where 
    D: Deserializer<'de>,
{
    Ok(
        match geo_types::Geometry::deserialize(deserializer)? {
            geo_types::Geometry::Polygon(v) => vec![v],
            geo_types::Geometry::MultiPolygon(v) => v.0,
            _ => vec![],
        }
    )
}

#[derive(serde::Deserialize, Debug)]
pub struct Country {
    name: String,
    #[serde(deserialize_with = "geojson::de::deserialize_geometry")]
    // geometry: Vec<geo_types::Polygon<f64>>,
    geometry: geo_types::Geometry,
    #[serde(skip)]
    color: Color,
}

pub fn generate_countries() -> Result<Vec<Country>> {
    use rand::prelude::*;

    Ok(geojson::de::deserialize_feature_collection_str_to_vec(&fs::read_to_string("low_res.geo.json")?)?)
}

pub fn test() -> Result<Vec<Country>> {
    let input_geojson = serde_json::json!(
        {
            "type":"FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": { "coordinates": [11.1,22.2], "type": "Point" },
                    "properties": {
                        "name": "Downtown",
                        "population": 123
                    }
                },
                {
                    "type": "Feature",
                    "geometry": { "coordinates": [33.3, 44.4], "type": "Point" },
                    "properties": {
                        "name": "Uptown",
                        "population": 456
                    }
                }
            ]
        }
    ).to_string();

    Ok(geojson::de::deserialize_feature_collection_str_to_vec(&input_geojson)?)
}


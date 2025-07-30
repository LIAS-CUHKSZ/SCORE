#include "helper.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::vector<double>> helper::readCSV(const std::string &filename)
{
  std::vector<std::vector<double>> data;
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open())
  {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return data;
  }

  // Skip header line
  std::getline(file, line);

  while (std::getline(file, line))
  {
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ','))
    {
      try
      {
        row.push_back(std::stod(cell));
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error parsing value: " << cell << std::endl;
        row.push_back(0.0);
      }
    }
    data.push_back(row);
  }

  file.close();
  return data;
}

// Function to perform semantic matching (simplified version)
void helper::matchLines(const std::vector<std::vector<double>> &lines2D,
                const std::vector<std::vector<double>> &lines3D,
                std::vector<int> &ids, std::vector<Eigen::Vector3d> &n_2D,
                std::vector<Eigen::Vector3d> &v_3D,
                std::vector<Eigen::Vector3d> &endpoints_3D)
{

  // Clear output vectors
  ids.clear();
  n_2D.clear();
  v_3D.clear();
  endpoints_3D.clear();

  // Count total matches first
  int total_matches = 0;
  for (size_t i = 0; i < lines2D.size(); i++)
  {
    double semantic_id_2d = lines2D[i][3]; // semantic_id column
    for (size_t j = 0; j < lines3D.size(); j++)
    {
      double semantic_id_3d = lines3D[j][6]; // semantic_id column
      if (std::abs(semantic_id_2d - semantic_id_3d) < 0.1)
      {
        total_matches++;
      }
    }
  }

  // Reserve space
  ids.reserve(total_matches);
  n_2D.reserve(total_matches);
  v_3D.reserve(total_matches);
  endpoints_3D.reserve(total_matches * 2);

  // Perform matching
  for (size_t i = 0; i < lines2D.size(); i++)
  {
    double semantic_id_2d = lines2D[i][3];
    Eigen::Vector3d normal_2d(lines2D[i][0], lines2D[i][1], lines2D[i][2]);

    for (size_t j = 0; j < lines3D.size(); j++)
    {
      double semantic_id_3d = lines3D[j][6];
      if (std::abs(semantic_id_2d - semantic_id_3d) < 0.1)
      {
        // Add match
        ids.push_back(i);
        n_2D.push_back(normal_2d);

        // Calculate 3D line direction vector
        Eigen::Vector3d p1(lines3D[j][0], lines3D[j][1], lines3D[j][2]);
        Eigen::Vector3d p2(lines3D[j][3], lines3D[j][4], lines3D[j][5]);
        Eigen::Vector3d direction = p2 - p1;
        direction.normalize();
        v_3D.push_back(direction);

        // Add endpoints
        endpoints_3D.push_back(p1);
        endpoints_3D.push_back(p2);
      }
    }
  }
}


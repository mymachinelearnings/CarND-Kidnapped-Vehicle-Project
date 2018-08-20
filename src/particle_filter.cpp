/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
/**
 * Initialize particle array with values around GPS co-ordinates
 * adding a random noise to each of x, y, theta
 */

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	num_particles = 30;

	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);

	for(int i=0; i<num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = x + dist_x(gen);
		p.y = y + dist_y(gen);
		p.theta = theta + dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;

	//cout <<"Init Complete!" << endl;

}

/**
 * This function predicts the next position of the particles
 * after time dt. Add measurement noises baesd around 0 mean and respective std_deviation
 *
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for(int i=0; i<num_particles; i++ ) {
		float thetaa = particles[i].theta;
		if(fabs(yaw_rate < 0.001)) {
			particles[i].x += velocity * delta_t * cos(thetaa);
			particles[i].y += velocity * delta_t * sin(thetaa);
		} else {
			particles[i].x += (velocity/yaw_rate) * (sin(thetaa + yaw_rate*delta_t) - sin(thetaa));
			particles[i].y += (velocity/yaw_rate) * ( cos(thetaa) - cos(thetaa + yaw_rate*delta_t));
		}

		particles[i].theta += yaw_rate * delta_t;


		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);


	}

	//cout <<"Predict Complete!" << endl;


}

/**
 * Process of matching our sensor Measurements to the objects in the real world - like Map landmarks
 * In layman terms, it means, suppose you get (12.3, 23.4, 21 radians). For this, you'll instead associate it with a nearest landmark measurement - like a lamp post instead of this raw data
 * This method loops through each observation
 * Each observation is compared with all the predictions(map landmarks in sensor range in this case)
 * to get the nearest neighbour of that map landmark
 * And finally assign it back to the observations list
 */
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//cout <<"Data Association Start!" << endl;

	for(unsigned int i=0; i<observations.size(); i++ ) {
		LandmarkObs obs = observations[i];

		double min_dist = numeric_limits<double>::max();
		int map_id = -1;

		//For each observation, loop through all the predicted values to get the NN
		for(unsigned int j=0; j<predicted.size(); j++) {
			LandmarkObs pred = predicted[j];

			double current_dist = dist(obs.x, obs.y, pred.x, pred.y);

			if(current_dist < min_dist) {
				min_dist = current_dist;
				map_id = pred.id;
			}
		}
		observations[i].id = map_id;
	}
	//cout <<"Data Association Complete!" << endl;

}

/**
 * Observations received here are from the sensor which are noisy and in car's perspective
 * First, convert them to map co-ordinate system
 * Next, associate each of them to the nearest map landmarks
 * Next, update the weights using the multi variate PDF
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	//cout <<"Update Weights Start!" << endl;

	double weight_normalizer = 0.0;


	for(unsigned int i=0; i<particles.size(); i++) {
		//cout <<"===============================================\n" << "Particle :: " << i << endl;
		double x_p = particles[i].x;
		double y_p = particles[i].y;
		double theta_p = particles[i].theta;




	/**
	 * For each particle, list through all the observations and transform them to map co-ordinates
	 * After this step, you'll have observations in map co-ordinate system for each particle
	 *
	 */
	vector<LandmarkObs> tr_observations;

	for(unsigned int j=0; j<observations.size(); j++) {
		//cout <<"Observation :: " << j << endl;
		LandmarkObs tr_ind_obs;

		tr_ind_obs.id = j;
		tr_ind_obs.x = x_p + (observations[j].x * cos(theta_p)) - (observations[j].y * sin(theta_p));
		tr_ind_obs.y = y_p + (observations[j].x * sin(theta_p)) + (observations[j].y * cos(theta_p));

		tr_observations.push_back(tr_ind_obs);

	}
	//cout <<"Transformation to Map co-or Complete!" << endl;

	/*
	 * You have observations.size() number of observations for each particle determined above
	 * Your observations will always be within the sensor range as the sensor cannot observe anything beyond its range
	 * But the map landmarks includes everything, but you really don't want to have all the landmarks for each particle
	 * just get the map landmarks within the sensor range, and leave the rest
	 *
	 */
	vector<LandmarkObs> pred_landmarks;
	for(unsigned int k=0; k<map_landmarks.landmark_list.size(); k++) {
		Map::single_landmark_s current_landmark = map_landmarks.landmark_list[k];

		if(fabs((current_landmark.x_f - x_p) <= sensor_range) && fabs((current_landmark.y_f - y_p) <= sensor_range )) {
			pred_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
		}
	}
	//cout <<"Total Predictions :::::::::::::: " << pred_landmarks.size() << endl;
	//cout <<"Predictions in Sensor Range Complete!" << endl;

	/*
	 * Determine the nearest neighbour of each observation form the landmarks in sensor range
	 *
	 */
	dataAssociation(pred_landmarks, tr_observations);
	//cout <<"NN Complete!" << endl;

	/**
	 * Calculating weight of each particle using Multivariate Gaussian PDF
	 * For each observation, calculate the PDF with predicted landmark position
	 * And finally multiply all the probabilities which gives the weight of a particular particle
	 */
	particles[i].weight = 1.0;

	//Measurement Noise parameters


    for(unsigned int l=0; l<tr_observations.size(); l++) {

    	double prob = 1.0;

    	double map_obs_id = tr_observations[l].id;
    	double map_obs_x = tr_observations[l].x;
    	double map_obs_y = tr_observations[l].y;

    	for(unsigned int m=0; m<pred_landmarks.size(); m++) {
    		double pred_id = pred_landmarks[m].id;
    		double pred_x = pred_landmarks[m].x;
    		double pred_y = pred_landmarks[m].y;

    		if(map_obs_id == pred_id) {
    		    double sigma_x = std_landmark[0];
    		    double sigma_y = std_landmark[1];

    			prob = (1.0/(2.0 * M_PI * sigma_x * sigma_y)) * exp(-1.0 * ((pow((pred_x - map_obs_x), 2)/(2.0 * sigma_x * sigma_x)) + (pow((pred_y - map_obs_y), 2)/(2.0 * sigma_y * sigma_y))));
    			particles[i].weight *= prob;
    		}

    	}
    }
    weight_normalizer += particles[i].weight;
	}

	//cout <<"Weight Normalizer :: " << weight_normalizer << endl;

	//Normalizing weights
	for (unsigned int n = 0; n < particles.size(); n++) {
		//cout <<"weight of particle " << n << particles[n].weight;
	    particles[n].weight /= weight_normalizer;
	    //cout <<"After normalization :: " << particles[n].weight;

	    weights[n] = particles[n].weight;
	    //cout <<"reassigning to weights complete!";
	}

	//cout <<"weight Update Complete!" << endl;

}

void ParticleFilter::resample() {

	vector<Particle> resampled_particles;

	  uniform_int_distribution<int> uniintdist(0, num_particles-1);
	  auto index = uniintdist(gen);

	  double max_weight = *max_element(weights.begin(), weights.end());

	  uniform_real_distribution<double> unirealdist(0.0, max_weight);

	  double beta = 0.0;

	  for (int i = 0; i < num_particles; i++) {
	    beta += unirealdist(gen) * 2.0;
	    while (beta > weights[index]) {
	      beta -= weights[index];
	      index = (index + 1) % num_particles;
	    }
	    resampled_particles.push_back(particles[index]);
	  }

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //TODO reset the values

	particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
